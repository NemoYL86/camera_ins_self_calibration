#ifndef OPENMVG_TRACKS_TRACKS_HPP
#define OPENMVG_TRACKS_TRACKS_HPP

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "data_typedef.h"
#include "flat_pair_map.h"
#include "union_find.h"

namespace hdmap {
namespace ddi {

// Data structure to store a track: collection of {ImageId,FeatureId}
//  The corresponding image points with their imageId and FeatureId.
using SubmapTrack = std::map<IndexT, IndexT>;
// A track is a collection of {trackId, SubmapTrack}
using Tracklets = std::map<IndexT, SubmapTrack>;

struct TracksBuilder {
    using IndexedFeaturePair = std::pair<IndexT, IndexT>;

    flat_pair_map<IndexedFeaturePair, IndexT> map_node_to_index;
    UnionFind uf_tree;

    /// Build tracks for a given series of pairWise matches
    void Build(const PairWiseMatches &map_pair_wise_matches)
    {
        // 1. We need to know how much single set we will have.
        //   i.e each set is made of a tuple : (imageIndex, featureIndex)
        std::set<IndexedFeaturePair> allFeatures;
        // For each couple of images list the used features
        for (const auto &iter : map_pair_wise_matches) {
            const auto &I = iter.first.first;
            const auto &J = iter.first.second;
            const std::vector<IndMatch> &vec_FilteredMatches = iter.second;

            // Retrieve all shared features and add them to a set
            for (const auto &cur_filtered_match : vec_FilteredMatches) {
                allFeatures.emplace(I, cur_filtered_match.i_);
                allFeatures.emplace(J, cur_filtered_match.j_);
            }
        }

        // 2. Build the 'flat' representation where a tuple (the node)
        //  is attached to a unique index.
        map_node_to_index.reserve(allFeatures.size());
        IndexT cpt = 0;
        for (const auto &feat : allFeatures) {
            map_node_to_index.emplace_back(feat, cpt);
            ++cpt;
        }
        // Sort the flat_pair_map
        map_node_to_index.sort();
        // Clean some memory
        allFeatures.clear();

        // 3. Add the node and the pairwise correpondences in the UF tree.
        uf_tree.InitSets(map_node_to_index.size());

        // 4. Union of the matched features corresponding UF tree sets
        for (const auto &iter : map_pair_wise_matches) {
            const auto &I = iter.first.first;
            const auto &J = iter.first.second;
            const std::vector<IndMatch> &vec_FilteredMatches = iter.second;
            for (const IndMatch &match : vec_FilteredMatches) {
                const IndexedFeaturePair pairI(I, match.i_);
                const IndexedFeaturePair pairJ(J, match.j_);
                // Link feature correspondences to the corresponding containing sets.
                uf_tree.Union(map_node_to_index[pairI], map_node_to_index[pairJ]);
            }
        }
    }

    /// Remove bad tracks (too short or track with ids collision)
    bool Filter(size_t nLengthSupTo = 2)
    {
        // Build the Track observations & mark tracks that have id collision:
        std::map<IndexT, std::set<IndexT>> tracks; // {track_id, {image_id, image_id, ...}}
        std::set<IndexT> problematic_track_id; // {track_id, ...}

        // For each node retrieve its track id from the UF tree and add the node to the track
        // - if an image id is observed multiple time, then mark the track as invalid
        //   - a track cannot list many times the same image index
        for (IndexT k = 0; k < map_node_to_index.size(); ++k) {
            const IndexT &track_id = uf_tree.Find(k);
            const auto &feat = map_node_to_index[k];

            // Augment the track and mark if invalid (an image can only be listed once)
            if (tracks[track_id].insert(feat.first.first).second == false) {
                problematic_track_id.insert(track_id); // invalid
            }
        }

        // Reject tracks that have too few observations
        for (const auto &val : tracks) {
            if (val.second.size() < nLengthSupTo) {
                problematic_track_id.insert(val.first);
            }
        }

        // Reset the marked invalid track ids in the UF Tree
        for (IndexT &root_index : uf_tree.m_cc_parent) {
            if (problematic_track_id.count(root_index) > 0) {
                // reset selected root
                uf_tree.m_cc_size[root_index] = 1;
                root_index = std::numeric_limits<IndexT>::max();
            }
        }
        return false;
    }

    /// Return the number of connected set in the UnionFind structure (tree forest)
    size_t NbTracks() const
    {
        std::set<IndexT> parent_id(uf_tree.m_cc_parent.begin(), uf_tree.m_cc_parent.end());
        // Erase the "special marker" that depicted rejected tracks
        parent_id.erase(std::numeric_limits<IndexT>::max());
        return parent_id.size();
    }

    /// Export tracks as a map (each entry is a sequence of imageId and featureIndex):
    ///  {TrackIndex => {(imageIndex, featureIndex), ... ,(imageIndex, featureIndex)}
    void ExportToSTL(Tracklets &map_tracks)
    {
        map_tracks.clear();
        for (IndexT k = 0; k < map_node_to_index.size(); ++k) {
            const auto &feat = map_node_to_index[k];
            const IndexT &track_id = uf_tree.m_cc_parent[k];
            if
                    (
                // ensure never add rejected elements (track marked as invalid)
                    track_id != std::numeric_limits<IndexT>::max()
                    // ensure never add 1-length track element (it's not a track)
                    && uf_tree.m_cc_size[track_id] > 1
                    ) {
                map_tracks[track_id].insert(feat.first);
            }
        }
    }
};

// This structure help to store the track visibility per view.
// Computing the tracks in common between many view can then be done
//  by computing the intersection of the track visibility for the asked view index.
// Thank to an additional array in memory this solution is faster than TracksUtilsMap::GetTracksInImages.
struct SharedTrackVisibilityHelper {
private:
    using TrackIdsPerView = std::map<IndexT, std::set<IndexT>>;

    TrackIdsPerView track_ids_per_view_;
    const Tracklets &tracks_;

public:

    explicit SharedTrackVisibilityHelper
            (
                    const Tracklets &tracks
            ) : tracks_(tracks)
    {
        for (const auto &tracks_it : tracks_) {
            // Add the track id visibility in the corresponding view track list
            for (const auto &track_obs_it : tracks_it.second) {
                track_ids_per_view_[track_obs_it.first].insert(tracks_it.first);
            }
        }
    }

    /**
     * @brief Find the shared tracks between some images ids.
     *
     * @param[in] image_ids: images id to consider
     * @param[out] tracks: tracks shared by the input images id
     */
    bool GetTracksInImages
            (
                    const std::set<IndexT> &image_ids,
                    Tracklets &tracks
            )
    {
        tracks.clear();
        if (image_ids.empty())
            return false;

        // Collect the shared tracks ids by the views
        std::set<IndexT> common_track_ids;
        {
            // Compute the intersection of all the track ids of the view's track ids.
            // 1. Initialize the track_id with the view first tracks
            // 2. Iteratively collect the common id of the remaining requested view
            auto image_index_it = image_ids.cbegin();
            if (track_ids_per_view_.count(*image_index_it)) {
                common_track_ids = track_ids_per_view_[*image_index_it];
            }
            bool merged = false;
            std::advance(image_index_it, 1);
            while (image_index_it != image_ids.cend()) {
                if (track_ids_per_view_.count(*image_index_it)) {
                    const auto ids_per_view_it = track_ids_per_view_.find(*image_index_it);
                    const auto &track_ids = ids_per_view_it->second;

                    std::set<IndexT> tmp;
                    std::set_intersection(
                            common_track_ids.cbegin(), common_track_ids.cend(),
                            track_ids.cbegin(), track_ids.cend(),
                            std::inserter(tmp, tmp.begin()));
                    common_track_ids.swap(tmp);
                    merged = true;
                }
                std::advance(image_index_it, 1);
            }
            if (image_ids.size() > 1 && !merged) {
                // If more than one image id is required and no merge operation have been done
                //  we need to reset the common track id
                common_track_ids.clear();
            }
        }

        // Collect the selected {img id, feat id} data for the shared track ids
        for (const auto track_ids_it : common_track_ids) {
            const auto track_it = tracks_.find(track_ids_it);
            const auto &track = track_it->second;
            // Find the corresponding output track and update it
            SubmapTrack &trackFeatsOut = tracks[track_it->first];
            for (const auto img_index: image_ids) {
                const auto track_view_info = track.find(img_index);
                trackFeatsOut[img_index] = track_view_info->second;
            }
        }
        return !tracks.empty();
    }
};

struct TracksUtilsMap {
    /**
     * @brief Find common tracks between images.
     *
     * @param[in] set_imageIndex: set of images we are looking for common tracks
     * @param[in] map_tracksIn: all tracks of the scene
     * @param[out] map_tracksOut: output with only the common tracks
     */
    static bool GetTracksInImages
            (
                    const std::set<IndexT> &set_imageIndex,
                    const Tracklets &map_tracksIn,
                    Tracklets &map_tracksOut
            )
    {
        map_tracksOut.clear();

        // Go along the tracks
        for (const auto &iterT : map_tracksIn) {
            // Look if the track contains the provided view index & save the point ids
            SubmapTrack map_temp;
            bool bTest = true;
            for (auto iterIndex = set_imageIndex.begin();
                 iterIndex != set_imageIndex.end() && bTest; ++iterIndex) {
                auto iterSearch = iterT.second.find(*iterIndex);
                if (iterSearch != iterT.second.end())
                    map_temp[iterSearch->first] = iterSearch->second;
                else
                    bTest = false;
            }

            if (!map_temp.empty() && map_temp.size() == set_imageIndex.size())
                map_tracksOut[iterT.first] = std::move(map_temp);
        }
        return !map_tracksOut.empty();
    }

    /// Return the tracksId as a set (sorted increasing)
    static void GetTracksIdVector
            (
                    const Tracklets &map_tracks,
                    std::set<IndexT> *set_tracksIds
            )
    {
        set_tracksIds->clear();
        for (const auto &iterT : map_tracks) {
            set_tracksIds->insert(iterT.first);
        }
    }

    /// Get feature index PerView and TrackId
    static bool GetFeatIndexPerViewAndTrackId
            (
                    const Tracklets &tracks,
                    const std::set<IndexT> &track_ids,
                    size_t nImageIndex,
                    std::vector<IndexT> *feat_ids
            )
    {
        feat_ids->reserve(tracks.size());
        for (const IndexT &trackId: track_ids) {
            const auto iterT = tracks.find(trackId);
            if (iterT != tracks.end()) {
                // Look if the desired image index exists in the track visibility
                const auto iterSearch = iterT->second.find(nImageIndex);
                if (iterSearch != iterT->second.end()) {
                    feat_ids->emplace_back(iterSearch->second);
                }
            }
        }
        return !feat_ids->empty();
    }

    /// Return the occurrence of tracks length.
    static void TracksLength
            (
                    const Tracklets &map_tracks,
                    std::map<IndexT, IndexT> &map_Occurence_TrackLength
            )
    {
        for (const auto &iterT : map_tracks) {
            const size_t trLength = iterT.second.size();
            if (map_Occurence_TrackLength.count(trLength) == 0) {
                map_Occurence_TrackLength[trLength] = 1;
            } else {
                map_Occurence_TrackLength[trLength] += 1;
            }
        }
    }

    /// Return a set containing the image Id considered in the tracks container.
    static void ImageIdInTracks
            (
                    const Tracklets &map_tracks,
                    std::set<IndexT> &set_imagesId
            )
    {
        for (const auto &iterT : map_tracks) {
            const SubmapTrack &map_ref = iterT.second;
            for (const auto &iter : map_ref) {
                set_imagesId.insert(iter.first);
            }
        }
    }
};

}
}

#endif // OPENMVG_TRACKS_TRACKS_HPP
