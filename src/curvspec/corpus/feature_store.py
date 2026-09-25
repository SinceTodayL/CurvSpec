import array
import ast
import os

import numpy as np


class FeatureStore:
    def __init__(self, data_directory):
        shape_path = os.path.join(data_directory, "shape.txt")
        with open(shape_path, "r", encoding="utf-8") as shape_file:
            self.nr_of_images, self.ndims = map(int, shape_file.readline().split())

        id_path = os.path.join(data_directory, "id.txt")
        with open(id_path, "rb") as id_file:
            names = id_file.read().strip().split()
        self.names = [name.decode("ISO-8859-1") for name in names]
        if len(self.names) != self.nr_of_images:
            raise ValueError("Feature id count does not match shape.txt")

        self.name2index = dict(zip(self.names, range(self.nr_of_images)))
        self.binary_file = os.path.join(data_directory, "feature.bin")
        print(f"[{self.__class__.__name__}] {self.nr_of_images}x{self.ndims} instances loaded from {data_directory}")

    def read(self, requested, is_name=True):
        requested = set(requested)
        if is_name:
            index_name_pairs = [(self.name2index[name], name) for name in requested if name in self.name2index]
        else:
            if requested and (min(requested) < 0 or max(requested) >= len(self.names)):
                raise IndexError("Feature index is out of range")
            index_name_pairs = [(index, self.names[index]) for index in requested]

        if not index_name_pairs:
            return [], []

        index_name_pairs.sort(key=lambda value: value[0])
        sorted_indices = [value[0] for value in index_name_pairs]
        offset = np.float32(1).nbytes * self.ndims
        values = array.array("f")

        with open(self.binary_file, "rb") as feature_file:
            feature_file.seek(index_name_pairs[0][0] * offset)
            values.fromfile(feature_file, self.ndims)
            previous = index_name_pairs[0][0]
            for next_index in sorted_indices[1:]:
                feature_file.seek((next_index - 1 - previous) * offset, 1)
                values.fromfile(feature_file, self.ndims)
                previous = next_index

        vectors = [
            values[index * self.ndims:(index + 1) * self.ndims].tolist()
            for index in range(len(index_name_pairs))
        ]
        return [value[1] for value in index_name_pairs], vectors

    def read_one(self, name):
        _, vectors = self.read([name])
        if not vectors:
            raise KeyError(f"Unknown feature id: {name}")
        return vectors[0]

    def shape(self):
        return [self.nr_of_images, self.ndims]


def read_mapping(path):
    with open(path, "r", encoding="utf-8") as mapping_file:
        value = ast.literal_eval(mapping_file.read())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return value
