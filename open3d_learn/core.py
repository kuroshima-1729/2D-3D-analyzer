import open3d.core as o3c
import numpy as np

import torch
import torch.utils.dlpack


def main():
    a = o3c.Tensor([0, 1, 2])
    print("Created from list:\n{}".format(a))
    a = o3c.Tensor(np.array([0, 1, 2]))
    print("\nCreated from numpy aray:\n{}".format(a))

    a_float = o3c.Tensor([0.0, 1.0, 2.0])
    print("\nDefault dtype and device:\n{}".format(a_float))

    a = o3c.Tensor(np.array([0, 1, 2]), dtype=o3c.Dtype.Float64)
    print("\nSpecified data type:\n{}".format(a))

    a = o3c.Tensor(np.array([0, 1, 2]), device=o3c.Device ("CUDA:0"))
    print("\nSpecified device:\n{}".format(a))

    np_a = np.ones((5,), dtype=np.int32)
    o3_a = o3c.Tensor(np_a)

    np_a[0] += 100
    o3_a[1] += 200
    print(f"np_a: {np_a}")
    print(f"o3_a: {o3_a}")

    o3_a = o3c.Tensor([1, 1, 1, 1, 1], dtype=o3c.Dtype.Int32)
    np_a = o3_a.numpy()

    np_a[0] += 100
    o3_a[1] += 200
    print(f"np_a: {np_a}")
    print(f"o3_a: {o3_a}")

    o3_a = o3c.Tensor([1, 1, 1, 1, 1], device=o3c.Device("CUDA:0"))
    print(f"\no3_a.cpu().numpy(): {o3_a.cpu().numpy()}")

    th_a = torch.ones((5, )).cuda(0)
    o3_a = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_a))
    print(f"th_a: {th_a}")
    print(f"o3_a: {o3_a}")

    th_a[0] = 100
    o3_a[1] = 200
    print(f"th_a: {th_a}")
    print(f"o3_a: {o3_a}")


if __name__ == "__main__":
    main()

