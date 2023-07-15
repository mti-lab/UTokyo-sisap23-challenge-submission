import re
import numpy as np

# from sklearn_extra.cluster import KMedoids
import faiss


class EPSearcher:
    def __init__(self, data: np.ndarray, cur_ep: int) -> None:
        self.data = data
        self.cur_ep = cur_ep

    def search(self, query: np.ndarray) -> int:
        raise NotImplementedError


class EPSearcherOriginal(EPSearcher):
    def __init__(self, data: np.ndarray, cur_ep: int) -> None:
        super().__init__(data, cur_ep)

    def search(self, query: np.ndarray) -> np.ndarray:
        return np.repeat(self.cur_ep, query.shape[0])


class EPSearcherRandom(EPSearcher):
    def __init__(self, data: np.ndarray, cur_ep: int, n_candidates: int) -> None:
        super().__init__(data, cur_ep)
        self.data_size = data.shape[0]
        # candidate ids
        self.cand_ids = np.random.randint(0, self.data_size, n_candidates)
        self.cand_ids = np.concatenate([self.cand_ids, [self.cur_ep]])
        self.cand_ids = np.unique(self.cand_ids)
        # search index
        self.index = faiss.IndexFlatL2(data.shape[1])
        self.index.add(data[self.cand_ids])

    def search(self, query: np.ndarray) -> np.ndarray:
        """Search the nearest entry point for each query.

        Args:
            query (np.ndarray): query batch (n_queries, d)

        Returns:
            np.ndarray: entry point ids (n_queries,)
        """
        _, I = self.index.search(query, 1)
        cid = I[:, 0]
        return self.cand_ids[cid]


class EPSearcherKmeans(EPSearcher):
    def __init__(self, data: np.ndarray, cur_ep: int, n_clusters: int) -> None:
        super().__init__(data, cur_ep)
        self.data_size = data.shape[0]

        d = data.shape[1]
        self.kmeans = faiss.Kmeans(d, n_clusters)
        self.kmeans.train(data)

        self.raw_index = faiss.IndexFlatL2(data.shape[1])
        self.raw_index.add(data)
        # cid -> nearest vector
        _, RI = self.raw_index.search(self.kmeans.centroids, 1)
        self.cand_ids = np.concatenate([RI[:, 0], [cur_ep]], axis=0)
        assert self.cand_ids.shape[0] == n_clusters + 1
        # search index
        self.index = faiss.IndexFlatL2(data.shape[1])
        self.index.add(data[self.cand_ids])

    def search(self, query: np.ndarray) -> np.ndarray:
        """Search the nearest entry point for each query.

        Args:
            query (np.ndarray): query batch (n_queries, d)

        Returns:
            np.ndarray: entry point ids (n_queries,)
        """
        _, I = self.kmeans.index.search(query, 1)
        cid = I[:, 0]
        return self.cand_ids[cid]
