#pragma once

#include <memory>
#include <vector>
#include <variant>
#include <optional>
#include <iostream>
#include <cassert>
#include <stack>
#include <iterator>
#include <cstddef>


template <typename K, typename V>
class Trie {

    // Since we inline build V's and return, need to be able to construct a default V.
    static_assert( std::is_default_constructible<V>() );

    using trie_search_data = typename std::optional<std::reference_wrapper<std::variant<std::monostate, V>>>;

protected:
    class TrieNode;

    class TrieNode {
    public:
        TrieNode() : parent_( nullptr ) { }

        std::vector<K> full_path_;
        std::vector<std::pair<K,std::unique_ptr<TrieNode>>> children_;
        std::variant<std::monostate, V> data_ptr_;
        TrieNode *parent_;
    };

    struct InternalSearchResult {
        InternalSearchResult( int iter_offset, std::reference_wrapper<std::unique_ptr<TrieNode>> landing_node ) :
            iter_offset_( iter_offset ), landing_node_( landing_node ) {}
        int iter_offset_;
        std::reference_wrapper<std::unique_ptr<TrieNode>> landing_node_;
    };

    std::unique_ptr<TrieNode> root_;
    size_t size_;

public:

    class TrieSearchResult {
    public:
        TrieSearchResult( const std::vector<K> &path, V& data ) :
            path_( path ), data_( data ) {}

        const std::vector<K> &path_;
        V &data_;
    };

    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = TrieNode;
        using pointer = TrieNode *;
        using reference = TrieNode&;

        Iterator( pointer ptr ) : m_ptr_( ptr ) {
            if( m_ptr_ == nullptr ) {
                return;
            }
            stack_.push( std::make_pair( 0, m_ptr_ ) );
            // Get to the first leaf
            while( !m_ptr_->children_.empty() ) {
                m_ptr_ = m_ptr_->children_[0].second.get();
                stack_.push( std::make_pair( 0, m_ptr_ ) );
            }
        }

        reference operator*() const {
            return *m_ptr_;
        }

        pointer operator->() {
            return m_ptr_;
        }

        Iterator &operator++() { 
            // Assume we are at a leaf node.
            assert( m_ptr_ == nullptr or m_ptr_->children_.empty() );
            if( m_ptr_ == nullptr ) {
                return *this;
            }
            stack_.pop(); // get us out of a leaf node

            for( ;; ) {
                if( stack_.empty() ) {
                    m_ptr_ = nullptr;
                    return *this;
                }
                m_ptr_ = stack_.top().second;
                if( m_ptr_->children_.empty() ) {
                    // Found a new leaf node!
                    return *this;
                }
                int &child_counter = stack_.top().first;

                // If we were just at 0,0,0 and popped, now we are at 0,1.
                child_counter++;
                if( (size_t) child_counter < m_ptr_->children_.size() ) {
                    pointer child = m_ptr_->children_[child_counter].second.get();
                    // The numbers mean "places we were just at."
                    // If we put a zero here, the code down below will assume that we have already seen
                    // the zeroth entry!
                    stack_.emplace( std::make_pair( -1, child ) );
                } else {
                    stack_.pop(); // up a level
                }
            }

        }
        Iterator operator++(int) { 
            return ++(*this);
        }

        friend bool operator== (const Iterator& a, const Iterator& b) {
            return a.m_ptr_ == b.m_ptr_;
        }

        friend bool operator!= (const Iterator& a, const Iterator& b) {
            return a.m_ptr_ != b.m_ptr_;
        }

    private:
        std::stack<std::pair<int,pointer>> stack_;
        pointer m_ptr_;
    };

    Trie() : root_( std::make_unique<TrieNode>() ), size_( 0 ) {
    }

    Iterator begin() { 
        Iterator iter( this->root_.get() );
        return iter;
    }
    Iterator end() { return Iterator( nullptr ); }

    // Search the path defined by key
    // Keep matching from the end of key going backwards until we can't match a K anymore.
    // Return a ref for the data_ptr entry at that point
    template <class it>
    trie_search_data search( it begin, it end ); 

    template <class it>
    std::optional<TrieSearchResult> search_any_subseq( it begin, it end );


    // Insert the path for key into the tree (if it isn't already)
    // Return a reference to V at that path (if exists) or default construct a V at the newly
    // constructed path and return a reference to it.
    template <class it>
    V &insert( it begin, it end );

    // Erase the path pointed to by key.
    // Collapse any existing paths that were created to disambiguate them from key.
    template <class it>
    void erase( it begin, it end );

    size_t size() {
        return size_;
    }

protected:
    template <class it>
    InternalSearchResult search_maximal_match( it begin, it end ) {

        static_assert( std::is_same<typename std::iterator_traits<it>::value_type, K>::value );

        auto node = std::ref( root_ );
        auto last_node = node;
        assert( root_ != nullptr );

        // Step 1: Go as far as we can.
        for( it pos = begin; pos != end; pos++ ) {
            last_node = node;
            
            // Find the child that matches our iteration key path
            for( auto &child_entry : node.get()->children_ ) {
                if( child_entry.first == *pos ) {
                    node = child_entry.second;
                    break;
                }
            }

            // No new match
            if( last_node.get() == node.get() ) {
                InternalSearchResult search_result( pos-begin, node );
                return search_result;
            }
        }

        InternalSearchResult search_result( end-begin, node );
        return search_result;
    }

};

template <class it, typename K>
bool is_same_path( std::vector<K> &full_path, it begin, it end, bool allow_subseq=false ) {

    static_assert( std::is_same<typename std::iterator_traits<it>::value_type, K>::value );
    auto iter_pos = begin;
    for( auto iter = full_path.begin(); iter != full_path.end(); iter++ ) {
        if( iter_pos == end ) {
            return false;
        }
        if( *iter_pos != *iter ) {
            return false;
        }
        iter_pos++;
    }
    return iter_pos == end or allow_subseq;
}

template <typename K, typename V>
template <class it>
typename Trie<K,V>::trie_search_data Trie<K,V>::search( it begin, it end ) {
    static_assert( std::is_same<typename std::iterator_traits<it>::value_type, K>::value );
    InternalSearchResult result = search_maximal_match( begin, end );

    std::unique_ptr<TrieNode> &landing_node = result.landing_node_.get();

    if( is_same_path( landing_node->full_path_, begin, end ) ) {

        return {std::ref( landing_node->data_ptr_ )};
    }

    // No match, nullopt
    return {};
}

#define likely(x) __builtin_expect( (x), 1 )

template <typename K, typename V>
template <class it>
std::optional<typename Trie<K,V>::TrieSearchResult> Trie<K,V>::search_any_subseq( it begin, it end ) {

    static_assert( std::is_same<typename std::iterator_traits<it>::value_type, K>::value );
    InternalSearchResult result = search_maximal_match( begin, end );

    std::unique_ptr<TrieNode> &landing_node = result.landing_node_.get();

    // This is a leaf! We matched a subseq!
    if( landing_node->children_.empty() and !landing_node->full_path_.empty() ) {
        // Check if whatever is here is a subseq of us.
        if( likely(result.iter_offset_ == (end-begin) || is_same_path( landing_node->full_path_, begin, end, true ) ) ) {
            assert( std::holds_alternative<V>( landing_node->data_ptr_ ) );
            return TrieSearchResult( landing_node->full_path_, std::get<V>( landing_node->data_ptr_ ) );
        } else {
            // No match, nullopt
            return {};
        }
    }

    // No match, nullopt
    return {};
}



template <typename K, typename V>
template <class it>
V &Trie<K,V>::insert( it begin, it end ) {
    static_assert( std::is_same<typename std::iterator_traits<it>::value_type, K>::value );

    InternalSearchResult result = search_maximal_match( begin, end );
    std::reference_wrapper<std::unique_ptr<TrieNode>> node_ref = result.landing_node_;
    auto iter = begin + result.iter_offset_;

    // If we match something there, return a V
    if( is_same_path( node_ref.get()->full_path_, begin, end ) ) {
        assert( std::holds_alternative<V>( node_ref.get()->data_ptr_ ) );
        return std::get<V>( node_ref.get()->data_ptr_ );
    }

    // We are for sure doing an insert. Count it
    size_++;

    // Otherwise, find out what is there
    auto &existing_path = node_ref.get()->full_path_;

    // Commandeer this (root) node
    if( node_ref.get()->children_.empty() and existing_path.empty() ) {
        existing_path = std::move( std::vector<K>( begin, end ) );
        node_ref.get()->data_ptr_ = V();
        return std::get<V>( node_ref.get()->data_ptr_ );
    }

    // We are child of this node 
    if( existing_path.empty() ) {
        assert( !node_ref.get()->children_.empty() );
        auto new_node = std::make_unique<TrieNode>();
        new_node->full_path_ = std::vector<K>( begin, end );
        new_node->data_ptr_ = V();
        new_node->parent_ = node_ref.get().get();
        std::pair<K, std::unique_ptr<TrieNode>> child( *iter, std::move( new_node ) );
        node_ref.get()->children_.emplace_back( std::move( child ) );
        return std::get<V>( node_ref.get()->children_.back().second->data_ptr_ );
    }

    // OK someone else is here too --- our keys differ.

    size_t cur_offset = (iter - begin);
    auto existing_path_iter = existing_path.begin() + (cur_offset);

    auto start_node = node_ref;

    for( ;; ) {
        auto new_node = std::make_unique<TrieNode>();
        new_node->parent_ = node_ref.get().get();


        // We found the spot that we differed. Make two unique nodes.
        if( *iter != *existing_path_iter ) {
            auto new_node2 = std::make_unique<TrieNode>();

            // Copy over the path and data pointer
            K existing_path_val = *existing_path_iter;
            new_node2->full_path_ = std::move( existing_path );
            new_node2->data_ptr_ = std::get<V>( start_node.get()->data_ptr_ );
            new_node2->parent_ = node_ref.get().get();

            // Set up our distinct node
            new_node->full_path_ = std::vector<K>( begin, end );
            new_node->data_ptr_ = V();
            //parent already set above.

            // Clear up old node memory
            node_ref.get()->data_ptr_ = std::monostate();

            // Set up children for node
            std::pair<K, std::unique_ptr<TrieNode>> child2( existing_path_val, std::move( new_node2 ) );
            node_ref.get()->children_.emplace_back( std::move( child2 ) );
            std::pair<K, std::unique_ptr<TrieNode>> child1( *iter, std::move( new_node ) );
            node_ref.get()->children_.emplace_back( std::move( child1 ) );

            return std::get<V>( node_ref.get()->children_.back().second->data_ptr_ );
        }

        // We are the same, keep going.
        std::pair<K, std::unique_ptr<TrieNode>> child( *iter, std::move( new_node ) );
        node_ref.get()->children_.emplace_back( std::move( child ) );
        iter++;
        existing_path_iter++;
        node_ref = node_ref.get()->children_.back().second;
    }

    assert( false );
    return std::get<V>( node_ref.get()->data_ptr_ );
}

template <typename K, typename V>
template <class it>
void Trie<K,V>::erase( it begin, it end ) {

    static_assert( std::is_same<typename std::iterator_traits<it>::value_type, K>::value );

    InternalSearchResult result = search_maximal_match( begin, end );
    std::reference_wrapper<std::unique_ptr<TrieNode>> found_node = result.landing_node_;

    // There is no key that matches
    if( !is_same_path( found_node.get()->full_path_, begin, end ) ) {
        return;
    }

    // We are for sure doing a delete. Count it
    size_--;


    if( result.landing_node_.get() == root_ ) {
        // We are the root!
        found_node.get()->full_path_.clear();
        assert( found_node.get()->children_.empty() );
        found_node.get()->data_ptr_ = std::monostate();
        return;
    }

    // OK so we aren't the root.
    // FIXME: Need doubly linked pointers here so we walk up to our parent.
    TrieNode *parent = found_node.get()->parent_;
    
    // The reason our node exists is because we differ from at least one sibling by a path entry.
    // That sibling must exist
    if( parent->children_.size() == 2 ) {

        // Need a temp ref for now... but find the sibling and bind it.
        std::reference_wrapper<std::unique_ptr<TrieNode>> sibling = found_node;
        for( auto iter = parent->children_.begin(); iter != parent->children_.end(); iter++ ) {
            if( iter->second != found_node.get() ) {
                sibling = std::ref( iter->second );
                break;
            }
        }

        // If the sibling is also a leaf, then there is no reason for this divergence to exist any longer.
        // Copy the sibling into the parent.
        if( sibling.get()->children_.empty() ) {
            parent->full_path_ = sibling.get()->full_path_;
            parent->data_ptr_ = sibling.get()->data_ptr_;

            // Since both the sibling (and found_ptr) are unique pointers in the parent's children, this will
            // delete both!
            parent->children_ = std::move( sibling.get()->children_ );

            // Now, walk back until we find a point where we diverge from someone else.
            TrieNode *node = parent;
            while( node->parent_ != nullptr ) {
                parent = node->parent_;
                if( parent->children_.size() >= 2 ) {
                    // Divergence point, we're done.
                    return;
                }
                parent->full_path_ = node->full_path_;
                parent->data_ptr_ = node->data_ptr_;
                parent->children_ = std::move( node->children_ );
                node = parent;
            }

            // We overwrote root, done!
            return;
        }

        // Fall through --- sibling is not a leaf
    }
    // remove us from parents children.
    for( auto iter = parent->children_.begin(); iter != parent->children_.end(); iter++ ) {
        if( iter->second == found_node.get() ) {
            parent->children_.erase( iter );
            return;
        }
    }
    // We weren't found? PANIC.
    assert( false );
}
