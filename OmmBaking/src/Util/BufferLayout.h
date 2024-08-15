// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <algorithm>
#include <assert.h>
#include <cstdint>

// Highest power of two that divides a given number
inline constexpr uint32_t highestPowerOf2( uint32_t n )
{
    return ( n & ( ~( n - 1 ) ) );
}

class BufferLayoutNode
{
public:

    enum Type
    {
        Type_None = 0,         // Node is either uninitialized, or has an external pointer and is unmaterialized.
        Type_Leaf = 1,         // Node is a leaf in the buffer layout tree.
        Type_Aggregate = 2,    // Inner node in tree. Child nodes are aggregated.
        Type_Overlay = 3,      // Inner node in tree. Child nodes are overlaid.
    };

    BufferLayoutNode()
        : m_type( Type_None )
        , m_hasParent( 0 )
        , m_isMaterialized( 0 )
        , m_alignSizePow2( 1 )
        , m_numBytes( 0 )
    {
    }   

    void attachChild( BufferLayoutNode* child )
    {
        assert( !m_isMaterialized && "Can't attach to a materialized node" );
        assert( !child->m_isMaterialized && "Can't attach materialized node" );
        assert( child->m_hasParent == 0 && "Child already has a parent, should be null" );

        if( m_childHead == nullptr )
        {
            // No prior children. Assign to both head and tail.
            assert( m_childTail == nullptr && "Tail pointer is not null when assigning first child." );
            m_childHead = child;
        }
        else
        {
            // Node already has children. Attach as sibling.
            assert( m_childTail != nullptr && "Tail pointer is null for node with children attached." );
            assert( m_childTail->m_sibling == nullptr && "Tail sibling pointer is not null." );
            m_childTail->m_sibling = child;
        }

        // Update tail to last entry.
        m_childTail = child;
        child->m_hasParent = 1;
        assert( child->m_sibling == nullptr && "Child already has a sibling, should be null" );
    }

    // returns a pointer one beyond this buffer node
    unsigned char* resolveLayout( unsigned char* ptr )
    {
        m_ptr = ptr;
        if( m_type == Type_Aggregate )
        {
            // track maximum alignment of aggregates.
            m_alignSizePow2 = 1;
            for( BufferLayoutNode* cs = m_childHead; cs; cs = cs->m_sibling )
            {
                ptr = cs->resolveLayout( ptr );
                m_alignSizePow2 = std::max( m_alignSizePow2, cs->m_alignSizePow2 );
            }
            m_numBytes = ptr - m_ptr;
        }
        else if( m_type == Type_Overlay )
        {
            // track maximum alignment of overlays.
            m_alignSizePow2 = 1;
            for( BufferLayoutNode* cs = m_childHead; cs; cs = cs->m_sibling )
            {
                unsigned char* end = cs->resolveLayout( ptr );
                ptr = std::max( ptr, end );
                m_alignSizePow2 = std::max( m_alignSizePow2, cs->m_alignSizePow2 );
            }
            m_numBytes = ptr - m_ptr;
        }
        else
        {
            // align buffer.
            m_ptr = ( unsigned char* )( ( ( uint64_t )ptr + m_alignSizePow2 - 1 ) & ~( m_alignSizePow2 - 1 ) );
        }

        m_isMaterialized = 1;

        return m_ptr + m_numBytes;
    }

    bool isMaterialized() const { return m_isMaterialized; };

    uint64_t        m_type           : 2;   // Node type, as defined by BufferLayout::Type
    uint64_t        m_hasParent      : 1;   // 0 if this is a root node
    uint64_t        m_isMaterialized : 1;   // True if resolveLayout has visited this node (called via BufferRef::materialize)
    uint64_t        m_alignSizePow2  : 8;   // Alignment in bytes for this buffer    
    uint64_t        m_numBytes       : 52;  // Represents the sub-tree size. If an extPtr is present and the tree is not yet materialized,
                                            // numBytes represents the size of the external buffer.
    
    unsigned char*  m_ptr = nullptr;

    BufferLayoutNode* m_childHead = nullptr; // First attached child (NULL if no children are attached)
    BufferLayoutNode* m_childTail = nullptr; // Last attached child (NULL if no children are attached)
    BufferLayoutNode* m_sibling   = nullptr; // Sibling node. Used when iterating over nodes in resolveLayout.
};

template<typename T=unsigned char>
class BufferLayout final : protected BufferLayoutNode
{
public:

    BufferLayout() { m_alignSizePow2 = highestPowerOf2( std::alignment_of<T>::value ); }
    ~BufferLayout() { }

    template <typename O>
    BufferLayout( const BufferLayout<O>& ) = delete;

    template <typename O>
    BufferLayout& operator=( const BufferLayout<O>& ) = delete;

    template <typename O>
    BufferLayout<T>& aggregate( BufferLayout<O>& child );   // Turns this buffer into an aggregate buffer, and adds the given buffer as a child.
                                                // Aggregate buffers represent memory regions where the child buffers are placed one after another.

    template <typename U, typename... Ts>
    BufferLayout<T>& aggregate( U& child0, Ts&... children ) // Variadic version of aggregate.
    {
        aggregate( child0 );
        aggregate( children... );
        return *this;
    }

    template <typename O>
    BufferLayout<T>& overlay( BufferLayout<O>& child );    // Turns this buffer into an overlay buffer, and adds the given buffer as a child.
                                               // Overlay buffers represent memory regions where the child buffers are overlaid on top of each other.

    template <typename U, typename... Ts>
    BufferLayout<T>& overlay( U& child0, Ts&... children ) // Variadic version of overlay.
    {
        overlay( child0 );
        overlay( children... );
        return *this;
    }

    T* access() const;

    BufferLayout& setNumElems( size_t numElems ) { return setNumBytes( numElems * sizeof( T ) ); }
    BufferLayout& setNumBytes( size_t numBytes );   // Forbidden on externals, aggregates, and overlays.
    BufferLayout<T>& setAlignmentInBytes( size_t alignmentInBytes );

    size_t getNumElems( void ) const { return getNumBytes() / sizeof( T ); }
    size_t getNumBytes( void ) const;
    size_t getAlignmentInBytes( void ) const { return m_alignSizePow2; };

    void materialize( T* ptr = 0 ); // Transitions the buffer to materialized state.

    bool isMaterialized() const { return BufferLayoutNode::isMaterialized(); };

    template<typename U>
    friend class BufferLayout;

};

//------------------------------------------------------------------------

template <typename T>
void BufferLayout<T>::materialize( T* ptr )
{
    assert( !m_hasParent && "May not directly materialize a child node" );
    resolveLayout( ( unsigned char* )ptr );
}

//------------------------------------------------------------------------

template <typename T> template <typename O>
BufferLayout<T>& BufferLayout<T>::aggregate( BufferLayout<O>& child )
{
    assert( ( m_type == BufferLayout::Type_Aggregate || m_type == BufferLayout::Type_None ) && "May only aggregate on Aggregate or None type nodes" );

    attachChild( &child );
    m_type = Type_Aggregate;

    return *this;
}

//------------------------------------------------------------------------

template <typename T> template <typename O>
BufferLayout<T>& BufferLayout<T>::overlay( BufferLayout<O>& child )
{
    assert( ( m_type == BufferLayout::Type_Overlay || m_type == BufferLayout::Type_None ) && "May only overlay on Overlay or None type nodes" );

    attachChild( &child );
    m_type = Type_Overlay;

    return *this;
}

//------------------------------------------------------------------------

template <typename T>
T* BufferLayout<T>::access() const
{
    if( m_numBytes == 0 )
        return nullptr;

    assert( m_isMaterialized && "Node must be materialized before accessing it" );
    unsigned char* ptr = m_ptr;

    assert( ( ( uintptr_t )ptr % alignof( T ) == 0 ) && "Sub-buffer is not properly aligned" );
    return ( T* )ptr;
}

//------------------------------------------------------------------------

template <typename T>
BufferLayout<T>& BufferLayout<T>::setNumBytes( size_t numBytes )
{
    assert( ( m_type == BufferLayout::Type_Leaf || m_type == BufferLayout::Type_None )&& "Only None and Leaf type nodes may be resized" );
    assert( !m_isMaterialized && "Can't resize materialized node" );
    assert( ( ( numBytes % sizeof( T ) ) == 0 ) && "New buffer size must be perfectly divisible by sizeof(T)" );

    m_type = Type_Leaf;
    m_numBytes = numBytes;
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
size_t BufferLayout<T>::getNumBytes( void ) const
{
    assert( ( m_isMaterialized || m_type == BufferLayout::Type_None || m_type == BufferLayout::Type_Leaf ) && "Can only query buffer size from materialized nodes, or unmaterialized None or Leaf type nodes." );
    assert( ( m_type != BufferLayout::Type_None || m_numBytes == 0 ) && "None type nodes must have a size of 0" );

    return m_numBytes;
}

//------------------------------------------------------------------------

template <typename T>
BufferLayout<T>& BufferLayout<T>::setAlignmentInBytes( size_t alignmentInBytes )
{
    assert( alignmentInBytes >= std::alignment_of<T>::value && "Alignment must be at least the data type size." );
    m_alignSizePow2 = highestPowerOf2( alignmentInBytes );
    return *this;
}