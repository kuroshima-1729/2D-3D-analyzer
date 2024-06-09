import open3d.core as o3c
import numpy as np



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

    np_a = np.oens((5,), dtypw=np.int32)
    o3_a = o3c.Tensor(np_a)


if __name__ == "__main__":
    main()

