#ifndef OPENMVG_SFM_SFM_DATA_HPP
#define OPENMVG_SFM_SFM_DATA_HPP

#include <string>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <memory>
#include <initializer_list>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/StdVector>

// Extend EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION with initializer list support.
#define EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_INITIALIZER_LIST(...)       \
namespace std {                                                            \
  template <>                                                              \
  class vector<__VA_ARGS__, std::allocator<__VA_ARGS__>>                   \
      : public vector<__VA_ARGS__, Eigen::aligned_allocator<__VA_ARGS__>> { \
    typedef vector<__VA_ARGS__, Eigen::aligned_allocator<__VA_ARGS__>>      \
        vector_base;                                                       \
                                                                           \
   public:                                                                 \
    typedef __VA_ARGS__ value_type;                                        \
    typedef vector_base::allocator_type allocator_type;                    \
    typedef vector_base::size_type size_type;                              \
    typedef vector_base::iterator iterator;                                \
    explicit vector(const allocator_type& a = allocator_type())            \
        : vector_base(a) {}                                                \
    template <typename InputIterator>                                      \
    explicit vector(InputIterator first, InputIterator last,                        \
           const allocator_type& a = allocator_type())                     \
        : vector_base(first, last, a) {}                                   \
    vector(const vector& c) = default;                                     \
    explicit vector(size_type num, const value_type& val = value_type())   \
        : vector_base(num, val) {}                                         \
    explicit vector(iterator start, iterator end) : vector_base(start, end) {}      \
    vector& operator=(const vector& x) = default;                          \
    /* Add initializer list constructor support*/                          \
    vector(std::initializer_list<__VA_ARGS__> list)                             \
        : vector_base(list.begin(), list.end()) {}                         \
  };                                                                       \
}  // namespace std

namespace hdmap {
namespace ddi {

/// Portable type used to store an index
using IndexT = uint64_t;

/// Portable value used to save an undefined index value
static const IndexT UndefinedIndexT = std::numeric_limits<IndexT>::max();

/// Standard Pair of IndexT
using Pair = std::pair<IndexT, IndexT>;

/// Set of Pair
using PairSet = std::set<Pair>;

/// Vector of Pair
using PairVec = std::vector<Pair>;

/**
* @brief Standard Hash_Map class
* @tparam K type of the keys
* @tparam V type of the values
*/
template<typename Key, typename Value>
using HashMap = std::map<Key, Value, std::less<Key>,
        Eigen::aligned_allocator<std::pair<const Key, Value>>>;

struct Feature {
    Feature()
    {}

    Feature(const Feature &feat)
    {
        landmarkId_ = feat.landmarkId_;
        cameraId_ = feat.cameraId_;
        frameId_ = feat.frameId_;
        uv_ = feat.uv_;
        xyz_ = feat.xyz_;
    }

    ~Feature()
    {}

    using Ptr = std::shared_ptr<Feature>;

    IndexT landmarkId_ = 0;
    IndexT cameraId_ = 0;
    IndexT frameId_ = 0;
    Eigen::Vector2d uv_ = {0, 0};
    Eigen::Vector3d xyz_ = {0, 0, 0};
};

struct Pose3d {
    Pose3d(uint64_t id, Eigen::Vector3d p, Eigen::Quaterniond q) :
            id_(id), p_(p), q_(q)
    {}

    Pose3d(uint64_t id, Eigen::Matrix4d pose) :
            id_(id)
    {
        FromPoseMat(pose);
    }

    Pose3d()
    {}

    ~Pose3d()
    {}

    uint64_t id_ = 0;
    Eigen::Vector3d p_ = {0, 0, 0};
    Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();

    Eigen::Matrix4d ToPoseMat() const
    {
        Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
        tf.topLeftCorner(3, 3) = q_.toRotationMatrix();
        tf.topRightCorner(3, 1) = p_;
        return tf;
    }

    void FromPoseMat(Eigen::Matrix4d pose)
    {
        p_ = pose.topRightCorner(3, 1);
        q_ = Eigen::Quaterniond(Eigen::Matrix3d(pose.topLeftCorner(3, 3)));
    }
};


struct IndMatch {
    IndMatch(const IndexT i = 0, const IndexT j = 0) : i_(i), j_(j)
    {}

    /// Remove duplicates ((i_, j_) that appears multiple times)
    static bool GetDeduplicated(std::vector<IndMatch> &vec_match)
    {
        const size_t sizeBefore = vec_match.size();
        const std::set<IndMatch> set_deduplicated(vec_match.cbegin(), vec_match.cend());
        vec_match.assign(set_deduplicated.cbegin(), set_deduplicated.cend());
        return sizeBefore != vec_match.size();
    }

    IndexT i_, j_;  // Left, right index
};

inline bool operator==(const IndMatch &m1, const IndMatch &m2)
{
    return (m1.i_ == m2.i_ && m1.j_ == m2.j_);
}

inline bool operator!=(const IndMatch &m1, const IndMatch &m2)
{
    return !(m1 == m2);
}

// Lexicographical ordering of matches. Used to remove duplicates
inline bool operator<(const IndMatch &m1, const IndMatch &m2)
{
    return (m1.i_ < m2.i_ || (m1.i_ == m2.i_ && m1.j_ < m2.j_));
}

inline std::ostream &operator<<(std::ostream &out, const IndMatch &obj)
{
    return out << obj.i_ << " " << obj.j_;
}

inline std::istream &operator>>(std::istream &in, IndMatch &obj)
{
    return in >> obj.i_ >> obj.j_;
}

using IndMatches = std::vector<IndMatch>;

/// Pairwise matches (indexed matches for a pair <I,J>)
/// The interface used to store corresponding point indexes per images pairs
class PairWiseMatchesContainer {
public:
    virtual ~PairWiseMatchesContainer() = default;

    virtual void insert(std::pair<Pair, IndMatches> &&pairWiseMatches) = 0;
};

//--
/// Pairwise matches (indexed matches for a pair <I,J>)
/// A structure used to store corresponding point indexes per images pairs
struct PairWiseMatches : public PairWiseMatchesContainer, public std::map<Pair, IndMatches> {
    void insert(std::pair<Pair, IndMatches> &&pairWiseMatches) override
    {
        std::map<Pair, IndMatches>::insert(
                std::forward<std::pair<Pair, IndMatches>>(pairWiseMatches));
    }
};

inline PairSet GetPairs(const PairWiseMatches &matches)
{
    PairSet pairs;
    for (const auto &cur_pair : matches) {
        pairs.insert(cur_pair.first);
    }
    return pairs;
}


}
}

#endif // OPENMVG_SFM_SFM_DATA_HPP
