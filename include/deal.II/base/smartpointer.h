// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 1998 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_smartpointer_h
#define dealii_smartpointer_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <atomic>
#include <typeinfo>

DEAL_II_NAMESPACE_OPEN

/**
 * This class represents an "observer pointer", i.e., it is a pointer class
 * like `std::unique_ptr` or `std::shared_ptr`, but it does not participate in
 * managing the lifetime of the object pointed to -- we are simply observing
 * an object passively that is owned by some other entity of the program, but
 * we are not in charge of creating and destroying this object. In particular,
 * the pointer may be pointing to a member variable of another object whose
 * lifetime is not actively managed but is instead tied to the lifetime of
 * the surrounding object of which it is a member.
 *
 * The class does, however, have mechanisms to ensure that the object pointed
 * to remains alive at least as long as the pointer points to -- in other words,
 * unlike a simple C-style `T*` variable, it can avoid the problem of
 * [dangling pointers](https://en.wikipedia.org/wiki/Dangling_pointer). In other
 * works, objects of the current type can be used like a regular
 * pointer (i.e., using the `*p` and `p->` operators and
 * through casting), but they make sure that the object pointed to is not
 * deleted or moved from in the course of use of the pointer by signaling the
 * pointee its use. This is achieved by keeping a use count for the pointed-to
 * object (which for this purpose needs to be derived from the Subscriptor
 * class), and ensuring that the pointed-to object's destructor triggers an
 * error if that use-count is larger than zero -- i.e., if there are still
 * observing SmartPointer objects pointing to it.
 *
 * Conceptually, SmartPointer fills a gap between `std::unique_ptr` and
 * `std::shared_ptr`. While the former makes it clear that there is a unique
 * owner of an object (namely the scope in which the `std::unique_ptr` resides),
 * it does not allow other places in a code base to point to the object. In
 * contrast, `std::shared_ptr` allows many places to point to the same object,
 * but none of them is the "owner" of the object: They all are, and the last
 * one to stop pointing to the object is responsible for deleting it.
 *
 * SmartPointer utilizes semantics in which one place owns an object and others
 * can point to it. The owning place is responsible for destroying the object
 * when it is no longer needed, and as mentioned above, this will trigger an
 * error if other places are still pointing to it via SmartPointer pointers.
 *
 * In practice, using this scheme, if you try to destroy an object to which
 * observers still point via SmartPointer objects, you will get an error that
 * says that there are still observers of the object and that the object can
 * consequently not be destroyed without creating "dangling" pointers. This
 * is often not very helpful in *finding* where these pointers are. As a
 * consequence, this class also provides two ways to annotate the observer
 * count with information about what other places still observe an object.
 * First, when initializing a SmartPointer object with the address of the
 * object pointed to, you can also attach a string that describes the
 * observing location, and this string will then be shown in error messages
 * listing all remaining observers. Second, if no such string is provided,
 * the name second template argument `P` is used as the debug string. This
 * allows to encode the observer information in the type of the SmartPointer.
 *
 * @note Unlike `std::unique_ptr` and `std::shared_ptr`, SmartPointer does
 *   NOT implement any memory handling. In particular, deleting a
 *   SmartPointer does not delete the object because the semantics
 *   of this class are that it only *observes* an object, but does not
 *   take over ownership. As a consequence, this is a sure way of creating
 *   a memory leak:
 *   @code
 *     SmartPointer<T> dont_do_this = new T;
 *   @endcode
 *   This is because here, no variable "owns" the object pointed to, and
 *   the destruction of the `dont_do_this` pointer does not trigger the
 *   release of the memory pointed to.
 *
 * @note This class correctly handles `const`-ness of an object, i.e.,
 *   a `SmartPointer<const T>` really behaves as if it were a pointer
 *   to a constant object (disallowing write access when dereferenced), while
 *   `SmartPointer<T>` is a mutable pointer.
 *
 * @ingroup memory
 */
template <typename T, typename P = void>
class SmartPointer
{
public:
  /**
   * Standard constructor for null pointer. The id of this pointer is set to
   * the name of the class P.
   */
  SmartPointer();

  /**
   * Copy constructor for SmartPointer. We do not copy the object subscribed
   * to from <tt>tt</tt>, but subscribe ourselves to it again.
   */
  template <class Q>
  SmartPointer(const SmartPointer<T, Q> &other);

  /**
   * Copy constructor for SmartPointer. We do not copy the object subscribed
   * to from <tt>tt</tt>, but subscribe ourselves to it again.
   */
  SmartPointer(const SmartPointer<T, P> &other);

  /**
   * Move constructor for SmartPointer.
   */
  SmartPointer(SmartPointer<T, P> &&other) noexcept;

  /**
   * Constructor taking a normal pointer. If possible, i.e. if the pointer is
   * not a null pointer, the constructor subscribes to the given object to
   * lock it, i.e. to prevent its destruction before the end of its use.
   *
   * The <tt>id</tt> is used in the call to Subscriptor::subscribe(id) and by
   * ~SmartPointer() in the call to Subscriptor::unsubscribe().
   */
  SmartPointer(T *t, const std::string &id);

  /**
   * Constructor taking a normal pointer. If possible, i.e. if the pointer is
   * not a null pointer, the constructor subscribes to the given object to
   * lock it, i.e. to prevent its destruction before the end of its use. The
   * id of this pointer is set to the name of the class P.
   */
  SmartPointer(T *t);

  /**
   * Destructor, removing the subscription.
   */
  ~SmartPointer();

  /**
   * Assignment operator for normal pointers. The pointer subscribes to the
   * new object automatically and unsubscribes to an old one if it exists. It
   * will not try to subscribe to a null-pointer, but still delete the old
   * subscription.
   */
  SmartPointer<T, P> &
  operator=(T *tt);

  /**
   * Assignment operator for SmartPointer. The pointer subscribes to the new
   * object automatically and unsubscribes to an old one if it exists.
   */
  template <class Q>
  SmartPointer<T, P> &
  operator=(const SmartPointer<T, Q> &other);

  /**
   * Assignment operator for SmartPointer. The pointer subscribes to the new
   * object automatically and unsubscribes to an old one if it exists.
   */
  SmartPointer<T, P> &
  operator=(const SmartPointer<T, P> &other);

  /**
   * Move assignment operator for SmartPointer.
   */
  SmartPointer<T, P> &
  operator=(SmartPointer<T, P> &&other) noexcept;

  /**
   * Delete the object pointed to and set the pointer to nullptr. Note
   * that unlike what the documentation of the class describes, *this
   * function actually deletes the object pointed to*. That is, this
   * function assumes a SmartPointer's ownership of the object pointed to.
   *
   * @deprecated This function is deprecated. It does not use the
   * semantics we usually use for this class, and its use is surely
   * going to be confusing.
   */
  DEAL_II_DEPRECATED
  void
  clear();

  /**
   * Conversion to normal pointer.
   */
  operator T *() const;

  /**
   * Dereferencing operator. This operator throws an ExcNotInitialized() if the
   * pointer is a null pointer.
   */
  T &
  operator*() const;

  /**
   * Return underlying pointer. This operator throws an ExcNotInitialized() if
   * the pointer is a null pointer.
   */
  T *
  get() const;

  /**
   * Operator that returns the underlying pointer. This operator throws an
   * ExcNotInitialized() if the pointer is a null pointer.
   */
  T *
  operator->() const;

  /**
   * Exchange the pointers of this object and the argument. Since both the
   * objects to which is pointed are subscribed to before and after, we do not
   * have to change their subscription counters.
   *
   * Note that this function (with two arguments) and the respective functions
   * where one of the arguments is a pointer and the other one is a C-style
   * pointer are implemented in global namespace.
   */
  template <class Q>
  void
  swap(SmartPointer<T, Q> &tt);

  /**
   * Swap pointers between this object and the pointer given. As this releases
   * the object pointed to presently, we reduce its subscription count by one,
   * and increase it at the object which we will point to in the future.
   *
   * Note that we indeed need a reference of a pointer, as we want to change
   * the pointer variable which we are given.
   */
  void
  swap(T *&ptr);

  /**
   * Return an estimate of the amount of memory (in bytes) used by this class.
   * Note in particular, that this only includes the amount of memory used by
   * <b>this</b> object, not by the object pointed to.
   */
  std::size_t
  memory_consumption() const;

private:
  /**
   * Pointer to the object we want to subscribe to. Since it is often
   * necessary to follow this pointer when debugging, we have deliberately
   * chosen a short name.
   */
  T *t;

  /**
   * The identification for the subscriptor.
   */
  const std::string id;

  /**
   * The Smartpointer is invalidated when the object pointed to is destroyed
   * or moved from.
   */
  std::atomic<bool> pointed_to_object_is_alive;
};


/* --------------------- inline Template functions ------------------------- */


template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer()
  : t(nullptr)
  , id(typeid(P).name())
  , pointed_to_object_is_alive(false)
{}



template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer(T *t)
  : t(t)
  , id(typeid(P).name())
  , pointed_to_object_is_alive(false)
{
  if (t != nullptr)
    t->subscribe(&pointed_to_object_is_alive, id);
}



template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer(T *t, const std::string &id)
  : t(t)
  , id(id)
  , pointed_to_object_is_alive(false)
{
  if (t != nullptr)
    t->subscribe(&pointed_to_object_is_alive, id);
}



template <typename T, typename P>
template <class Q>
inline SmartPointer<T, P>::SmartPointer(const SmartPointer<T, Q> &other)
  : t(other.t)
  , id(other.id)
  , pointed_to_object_is_alive(false)
{
  if (other != nullptr)
    {
      Assert(other.pointed_to_object_is_alive,
             ExcMessage("You can't copy a smart pointer object that "
                        "is pointing to an object that is no longer alive."));
      t->subscribe(&pointed_to_object_is_alive, id);
    }
}



template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer(const SmartPointer<T, P> &other)
  : t(other.t)
  , id(other.id)
  , pointed_to_object_is_alive(false)
{
  if (other != nullptr)
    {
      Assert(other.pointed_to_object_is_alive,
             ExcMessage("You can't copy a smart pointer object that "
                        "is pointing to an object that is no longer alive."));
      t->subscribe(&pointed_to_object_is_alive, id);
    }
}



template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer(SmartPointer<T, P> &&other) noexcept
  : t(other.t)
  , id(other.id)
  , pointed_to_object_is_alive(false)
{
  if (other != nullptr)
    {
      Assert(other.pointed_to_object_is_alive,
             ExcMessage("You can't move a smart pointer object that "
                        "is pointing to an object that is no longer alive."));

      try
        {
          t->subscribe(&pointed_to_object_is_alive, id);
        }
      catch (...)
        {
          Assert(false,
                 ExcMessage(
                   "Calling subscribe() failed with an exception, but we "
                   "are in a function that cannot throw exceptions. "
                   "Aborting the program here."));
        }

      // Release the rhs object as if we had moved all members away from
      // it directly:
      other = nullptr;
    }
}



template <typename T, typename P>
inline SmartPointer<T, P>::~SmartPointer()
{
  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(&pointed_to_object_is_alive, id);
}



template <typename T, typename P>
inline void
SmartPointer<T, P>::clear()
{
  if (pointed_to_object_is_alive && t != nullptr)
    {
      t->unsubscribe(&pointed_to_object_is_alive, id);
      delete t;
      Assert(pointed_to_object_is_alive == false, ExcInternalError());
    }
  t = nullptr;
}



template <typename T, typename P>
inline SmartPointer<T, P> &
SmartPointer<T, P>::operator=(T *tt)
{
  // optimize if no real action is requested
  if (t == tt)
    return *this;

  // Let us unsubscribe from the current object
  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(&pointed_to_object_is_alive, id);

  // Then reset to the new object, and subscribe to it
  t = tt;
  if (tt != nullptr)
    t->subscribe(&pointed_to_object_is_alive, id);

  return *this;
}



template <typename T, typename P>
template <class Q>
inline SmartPointer<T, P> &
SmartPointer<T, P>::operator=(const SmartPointer<T, Q> &other)
{
  // if objects on the left and right
  // hand side of the operator= are
  // the same, then this is a no-op
  if (&other == this)
    return *this;

  // Let us unsubscribe from the current object
  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(&pointed_to_object_is_alive, id);

  // Then reset to the new object, and subscribe to it
  t = (other != nullptr ? other.get() : nullptr);
  if (other != nullptr)
    {
      Assert(other.pointed_to_object_is_alive,
             ExcMessage("You can't copy a smart pointer object that "
                        "is pointing to an object that is no longer alive."));
      t->subscribe(&pointed_to_object_is_alive, id);
    }
  return *this;
}



template <typename T, typename P>
inline SmartPointer<T, P> &
SmartPointer<T, P>::operator=(const SmartPointer<T, P> &other)
{
  // if objects on the left and right
  // hand side of the operator= are
  // the same, then this is a no-op
  if (&other == this)
    return *this;

  // Let us unsubscribe from the current object
  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(&pointed_to_object_is_alive, id);

  // Then reset to the new object, and subscribe to it
  t = (other != nullptr ? other.get() : nullptr);
  if (other != nullptr)
    {
      Assert(other.pointed_to_object_is_alive,
             ExcMessage("You can't copy a smart pointer object that "
                        "is pointing to an object that is no longer alive."));
      t->subscribe(&pointed_to_object_is_alive, id);
    }
  return *this;
}



template <typename T, typename P>
inline SmartPointer<T, P> &
SmartPointer<T, P>::operator=(SmartPointer<T, P> &&other) noexcept
{
  if (other == nullptr)
    {
      *this = nullptr;
    }
  // if objects on the left and right hand side of the operator= are
  // the same, then this is a no-op
  else if (&other != this)
    {
      // Let us unsubscribe from the current object
      if (t != nullptr && pointed_to_object_is_alive)
        t->unsubscribe(&pointed_to_object_is_alive, id);

      // Then reset to the new object, and subscribe to it:
      Assert(other.pointed_to_object_is_alive,
             ExcMessage("You can't move a smart pointer object that "
                        "is pointing to an object that is no longer alive."));
      t = other.get();
      try
        {
          t->subscribe(&pointed_to_object_is_alive, id);
        }
      catch (...)
        {
          Assert(false,
                 ExcMessage(
                   "Calling subscribe() failed with an exception, but we "
                   "are in a function that cannot throw exceptions. "
                   "Aborting the program here."));
        }

      // Finally release the rhs object since we moved its contents
      other = nullptr;
    }
  return *this;
}



template <typename T, typename P>
inline SmartPointer<T, P>::operator T *() const
{
  return t;
}



template <typename T, typename P>
inline T &
SmartPointer<T, P>::operator*() const
{
  Assert(t != nullptr, ExcNotInitialized());
  Assert(pointed_to_object_is_alive,
         ExcMessage("The object pointed to is not valid anymore."));
  return *t;
}



template <typename T, typename P>
inline T *
SmartPointer<T, P>::get() const
{
  Assert(t != nullptr, ExcNotInitialized());
  Assert(pointed_to_object_is_alive,
         ExcMessage("The object pointed to is not valid anymore."));
  return t;
}



template <typename T, typename P>
inline T *
SmartPointer<T, P>::operator->() const
{
  return this->get();
}



template <typename T, typename P>
template <class Q>
inline void
SmartPointer<T, P>::swap(SmartPointer<T, Q> &other)
{
#ifdef DEBUG
  SmartPointer<T, P> aux(t, id);
  *this = other;
  other = aux;
#else
  std::swap(t, other.t);
#endif
}



template <typename T, typename P>
inline void
SmartPointer<T, P>::swap(T *&ptr)
{
  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(pointed_to_object_is_alive, id);

  std::swap(t, ptr);

  if (t != nullptr)
    t->subscribe(pointed_to_object_is_alive, id);
}



template <typename T, typename P>
inline std::size_t
SmartPointer<T, P>::memory_consumption() const
{
  return sizeof(SmartPointer<T, P>);
}



// The following function is not strictly necessary but is an optimization
// for places where you call swap(p1,p2) with SmartPointer objects p1, p2.
// Unfortunately, MS Visual Studio (at least up to the 2013 edition) trips
// over it when calling std::swap(v1,v2) where v1,v2 are std::vectors of
// SmartPointer objects: it can't determine whether it should call std::swap
// or dealii::swap on the individual elements (see bug #184 on our Google Code
// site. Consequently, just take this function out of the competition for this
// compiler.
#ifndef _MSC_VER
/**
 * Global function to swap the contents of two smart pointers. As both objects
 * to which the pointers point retain to be subscribed to, we do not have to
 * change their subscription count.
 */
template <typename T, typename P, class Q>
inline void
swap(SmartPointer<T, P> &t1, SmartPointer<T, Q> &t2)
{
  t1.swap(t2);
}
#endif


/**
 * Global function to swap the contents of a smart pointer and a C-style
 * pointer.
 *
 * Note that we indeed need a reference of a pointer, as we want to change the
 * pointer variable which we are given.
 */
template <typename T, typename P>
inline void
swap(SmartPointer<T, P> &t1, T *&t2)
{
  t1.swap(t2);
}



/**
 * Global function to swap the contents of a C-style pointer and a smart
 * pointer.
 *
 * Note that we indeed need a reference of a pointer, as we want to change the
 * pointer variable which we are given.
 */
template <typename T, typename P>
inline void
swap(T *&t1, SmartPointer<T, P> &t2)
{
  t2.swap(t1);
}

DEAL_II_NAMESPACE_CLOSE

#endif
