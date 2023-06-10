check_if_overlap = (
    lambda block_A, block_B: block_B[0] <= block_A[0] <= block_B[1]
    or block_A[0] <= block_B[0] <= block_A[1]
)


def compute_common(values):
    basis = values[0]

    for OBS_data in values[1:]:

        new_basis = []

        for base_index, base in enumerate(basis):
            for wave_block in OBS_data:
                if check_if_overlap(base, wave_block):
                    new_basis.append(base.copy())

                    new_basis[-1][0] = max(wave_block[0], base[0])
                    new_basis[-1][1] = min(wave_block[1], base[1])

            basis = new_basis
    return basis


if __name__ == "__main__":
    others = [[[1, 6.1], [7, 10]], [[0, 4], [6, 8], [9, 11]], [[1, 12.2]]]
    out = compute_common(others)
    print("fOUND -> ", out)
    import matplotlib.pyplot as plt

    for entry in out:
        print(entry)
        plt.plot(entry, (1, 1))
        plt.axvline(entry[0], color="red", linestyle="--")
        plt.axvline(entry[1], color="red", linestyle="--")

    for obs_index, OBSentry in enumerate(others):
        for entry in OBSentry:
            off = 0.1 * obs_index
            plt.plot(entry, (0 + off, off + 0))
