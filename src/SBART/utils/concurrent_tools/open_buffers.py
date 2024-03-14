from multiprocessing import shared_memory

from numpy import ndarray


def open_buffer(buffer_info: dict, open_type: str, buffers: list):
    if open_type == "template":
        open_order = [
            "template",
            "template_errors",
            "template_wavelength",
            "template_counts",
        ]
    elif open_type == "BayesianCache":
        open_order = ["mask_cache", "cached_orders"]
    else:
        raise Exception(f"{open_type=} does not exist")

    data = []

    try:
        for array_name in open_order:
            shm = shared_memory.SharedMemory(name=buffer_info[array_name][0].name)
            data_array = ndarray(
                buffer_info[array_name][1],
                dtype=buffer_info[array_name][2],
                buffer=shm.buf,
            )
            buffers.append(shm)
            data.append(data_array)

        return *data, buffers

    except Exception as e:
        print(f"Opening buffers failed in  {array_name=}  due to {e}")
        print("Available buffers: ", buffer_info.keys(), "worker_outputs" in buffer_info)
        for buff in buffers:
            buff.close()
        raise Exception
