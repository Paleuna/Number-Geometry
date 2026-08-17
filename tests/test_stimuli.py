from number_geometry.stimuli import generate_unified_dataset


def test_handcrafted_dataset_size_and_targets():
    data = generate_unified_dataset()
    assert len(data) == 990
    for item in data[:100]:
        assert item["text"].split()[item["target_idx"]].strip(".,:!?\"'") == item["target_token"]
