"""Small datasets must keep every explicitly padded distributed rank active."""

from itertools import islice

from cosmos_framework.data.generator.dataflow.distributors import MapDistributor


class Samples:
    def __len__(self):
        return 5

    def get_exposure(self, index, exposure):
        return {"sample": index, "exposure": exposure}


def test_five_samples_eight_ranks_repeat_with_distinct_exposures():
    distributor = MapDistributor(Samples(), shuffle=False, pad_to_world_size=True)
    streams = [list(islice(distributor.stream(rank, 8, 0, 1), 5)) for rank in range(8)]
    assert {item["sample"] for stream in streams for item in stream} == set(range(5))
    assert {item["exposure"] for stream in streams for item in stream} == set(range(40))
    for rank, stream in enumerate(streams):
        assert [item["sample"] for item in stream] == [rank % 5] * 5
        assert [item["_dp_epoch"] for item in stream] == list(range(5))
        assert [item["_dp_stream_pos"] for item in stream] == [0] * 5


def test_default_does_not_silently_repeat_small_datasets():
    distributor = MapDistributor(Samples(), shuffle=False)
    assert list(distributor.stream(5, 8, 0, 1)) == []


def test_padding_resume_restores_next_exposure(monkeypatch):
    monkeypatch.setenv("COSMOS_DL_STATE_WORKER_0_EPOCH", "2")
    monkeypatch.setenv("COSMOS_DL_STATE_WORKER_0_INDEX", "0")
    distributor = MapDistributor(Samples(), shuffle=False, pad_to_world_size=True)
    item = next(distributor.stream(7, 8, 0, 1))
    assert (item["sample"], item["exposure"], item["_dp_epoch"]) == (2, 31, 3)
