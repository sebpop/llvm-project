! RUN: %flang_fc1 -emit-llvm %s -o - | opt -passes='print<delinearization>' -disable-output 2>&1 | FileCheck %s

! Check that delinearization successfully recognizes the 2d and 3d array structures.

! CHECK: Printing analysis 'Delinearization' for function '{{.*}}test_2d_array_access{{.*}}':
! CHECK: ArrayDecl[20][10] with elements of 4 bytes
! CHECK: ArrayRef[{{.*}}][{{.*}}]

! CHECK: Printing analysis 'Delinearization' for function '{{.*}}test_3d_array_access{{.*}}':
! CHECK: ArrayDecl[15][10][5] with elements of 4 bytes
! CHECK: ArrayRef[{{.*}}][{{.*}}][{{.*}}]

module array_test
  implicit none

  interface
    function get_index() result(idx)
      integer :: idx
    end function get_index
  end interface

contains

subroutine test_2d_array_access()
    implicit none
    integer :: arr(10, 20)
    integer :: i, j, idx, sum

    idx = get_index()
    sum = 0

    do i = 1, 5
        do j = 1, 5
            arr(i, j) = idx + i * j
            sum = sum + arr(i, j)
        end do
    end do

    do i = 1, 3
        j = mod(idx + i, 20) + 1
        arr(i, j) = sum + i
    end do

    print *, sum
    call external_use_2d(arr)
end subroutine test_2d_array_access

subroutine test_3d_array_access()
    implicit none
    integer :: arr(5, 10, 15)
    integer :: i, j, k, idx, sum

    idx = get_index()
    sum = 0

    do k = 1, 3
        do j = 1, 4
            do i = 1, 2
                arr(i, j, k) = idx + i * j * k
                sum = sum + arr(i, j, k)
            end do
        end do
    end do

    do i = 1, 2
        j = mod(idx + i, 10) + 1
        k = mod(idx + i * 2, 15) + 1
        arr(i, j, k) = sum + i * j
    end do

    print *, sum
    call external_use_3d(arr)
end subroutine test_3d_array_access

end module array_test

subroutine external_use_2d(arr)
  integer, intent(in) :: arr(10, 20)
  ! Do nothing, just prevent optimization.
end subroutine external_use_2d

subroutine external_use_3d(arr)
  integer, intent(in) :: arr(5, 10, 15)
  ! Do nothing, just prevent optimization.
end subroutine external_use_3d
