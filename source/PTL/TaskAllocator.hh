//
// MIT License
// Copyright (c) 2018 Jonathan R. Madsen
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//
// ------------------------------------------------------------
// Tasking class header file
//
// Class Description:
//
// A class for fast allocation of objects to the heap through a pool of
// chunks organised as linked list. It's meant to be used by associating
// it to the object to be allocated and defining for it new and delete
// operators via MallocSingle() and FreeSingle() methods.

//      ---------------- TaskAllocator ----------------
//
// Author: G.Cosmo (CERN), November 2000
// ------------------------------------------------------------

#ifndef TaskAllocator_h
#define TaskAllocator_h 1

#include <cstddef>
#include <typeinfo>

#include "PTL/TaskAllocatorPool.hh"

class TaskAllocatorBase
{
public:
    TaskAllocatorBase();
    virtual ~TaskAllocatorBase();
    virtual void ResetStorage()=0;
    virtual size_t GetAllocatedSize() const=0;
    virtual int GetNoPages() const=0;
    virtual size_t GetPageSize() const=0;
    virtual void IncreasePageSize( unsigned int sz )=0;
    virtual const char* GetPoolType() const=0;
};

template <class Type>
class TaskAllocator : public TaskAllocatorBase
{
public:  // with description

    TaskAllocator();
    ~TaskAllocator();
    // Constructor & destructor

    inline Type* MallocSingle();
    inline void FreeSingle(Type* anElement);
    // Malloc and Free methods to be used when overloading
    // new and delete operators in the client <Type> object

    inline void ResetStorage();
    // Returns allocated storage to the free store, resets allocator.
    // Note: contents in memory are lost using this call !

    inline size_t GetAllocatedSize() const;
    // Returns the size of the total memory allocated
    inline int GetNoPages() const;
    // Returns the total number of allocated pages
    inline size_t GetPageSize() const;
    // Returns the current size of a page
    inline void IncreasePageSize( unsigned int sz );
    // Resets allocator and increases default page size of a given factor

    inline const char* GetPoolType() const;
    // Returns the type_info Id of the allocated type in the pool

public:  // without description

    // This public section includes standard methods and types
    // required if the allocator is to be used as alternative
    // allocator for STL containers.
    // NOTE: the code below is a trivial implementation to make
    //       this class an STL compliant allocator.
    //       It is anyhow NOT recommended to use this class as
    //       alternative allocator for STL containers !

    typedef Type value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef Type* pointer;
    typedef const Type* const_pointer;
    typedef Type& reference;
    typedef const Type& const_reference;

    template <class U> TaskAllocator(const TaskAllocator<U>& right) throw()
        : mem(right.mem) {}
    // Copy constructor

    pointer address(reference r) const { return &r; }
    const_pointer address(const_reference r) const { return &r; }
    // Returns the address of values

    pointer allocate(size_type n, void* = 0)
    {
        // Allocates space for n elements of type Type, but does not initialise
        //
        Type* mem_alloc = 0;
        if (n == 1)
            mem_alloc = MallocSingle();
        else
            mem_alloc = static_cast<Type*>(::operator new(n*sizeof(Type)));
        return mem_alloc;
    }
    void deallocate(pointer p, size_type n)
    {
        // Deallocates n elements of type Type, but doesn't destroy
        //
        if (n == 1)
            FreeSingle(p);
        else
            ::operator delete((void*)p);
        return;
    }

    void construct(pointer p, const Type& val) { new((void*)p) Type(val); }
    // Initialises *p by val
    void destroy(pointer p) { p->~Type(); }
    // Destroy *p but doesn't deallocate

    size_type max_size() const throw()
    {
        // Returns the maximum number of elements that can be allocated
        //
        return 2147483647/sizeof(Type);
    }

    template <class U>
    struct rebind { typedef TaskAllocator<U> other; };
    // Rebind allocator to type U

    TaskAllocatorPool mem;
    // Pool of elements of sizeof(Type)

private:

    const char* tname;
    // Type name identifier
};

// ------------------------------------------------------------
// Inline implementation
// ------------------------------------------------------------

// Initialization of the static pool
//
// template <class Type> TaskAllocatorPool TaskAllocator<Type>::mem(sizeof(Type));

// ************************************************************
// TaskAllocator constructor
// ************************************************************
//
template <class Type>
TaskAllocator<Type>::TaskAllocator()
: mem(sizeof(Type))
{
    tname = typeid(Type).name();
}

// ************************************************************
// TaskAllocator destructor
// ************************************************************
//
template <class Type>
TaskAllocator<Type>::~TaskAllocator()
{ }

// ************************************************************
// MallocSingle
// ************************************************************
//
template <class Type>
Type* TaskAllocator<Type>::MallocSingle()
{
    return static_cast<Type*>(mem.Alloc());
}

// ************************************************************
// FreeSingle
// ************************************************************
//
template <class Type>
void TaskAllocator<Type>::FreeSingle(Type* anElement)
{
    mem.Free(anElement);
    return;
}

// ************************************************************
// ResetStorage
// ************************************************************
//
template <class Type>
void TaskAllocator<Type>::ResetStorage()
{
    // Clear all allocated storage and return it to the free store
    //
    mem.Reset();
    return;
}

// ************************************************************
// GetAllocatedSize
// ************************************************************
//
template <class Type>
size_t TaskAllocator<Type>::GetAllocatedSize() const
{
    return mem.Size();
}

// ************************************************************
// GetNoPages
// ************************************************************
//
template <class Type>
int TaskAllocator<Type>::GetNoPages() const
{
    return mem.GetNoPages();
}

// ************************************************************
// GetPageSize
// ************************************************************
//
template <class Type>
size_t TaskAllocator<Type>::GetPageSize() const
{
    return mem.GetPageSize();
}

// ************************************************************
// IncreasePageSize
// ************************************************************
//
template <class Type>
void TaskAllocator<Type>::IncreasePageSize( unsigned int sz )
{
    ResetStorage();
    mem.GrowPageSize(sz);
}

// ************************************************************
// GetPoolType
// ************************************************************
//
template <class Type>
const char* TaskAllocator<Type>::GetPoolType() const
{
    return tname;
}

// ************************************************************
// operator==
// ************************************************************
//
template <class T1, class T2>
bool operator== (const TaskAllocator<T1>&, const TaskAllocator<T2>&) throw()
{
    return true;
}

// ************************************************************
// operator!=
// ************************************************************
//
template <class T1, class T2>
bool operator!= (const TaskAllocator<T1>&, const TaskAllocator<T2>&) throw()
{
    return false;
}

#endif
