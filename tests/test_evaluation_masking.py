import numpy as np
import torch


def apply_known_positive_mask(rating, train_pos, batch_users, extra_exclude_dict=None):
    exclude_index, exclude_items = [], []
    for row, (uid_raw, items) in enumerate(zip(batch_users, train_pos)):
        known_items = set(int(x) for x in items)
        if extra_exclude_dict is not None:
            uid = int(np.atleast_1d(uid_raw)[0])
            known_items.update(int(x) for x in extra_exclude_dict.get(uid, []))
        exclude_index.extend([row] * len(known_items))
        exclude_items.extend(known_items)
    if exclude_items:
        rating[exclude_index, exclude_items] = -(1 << 10)
    return rating


def test_validation_final_test_masks_train_and_validation_positives():
    rating = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    train_pos = [np.array([1, 2]), np.array([0])]
    val_pos = {10: [3], 20: [4, 5]}
    out = apply_known_positive_mask(rating.clone(), train_pos, [10, 20], val_pos)
    for row, items in enumerate(([1, 2, 3], [0, 4, 5])):
        for item in items:
            assert out[row, item].item() == -(1 << 10)


def test_fixed_epoch_mode_masks_training_positives_only():
    rating = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    out = apply_known_positive_mask(rating.clone(), [np.array([1, 2])], [10], None)
    assert out[0, 1].item() == -(1 << 10)
    assert out[0, 2].item() == -(1 << 10)
    assert out[0, 3].item() == rating[0, 3].item()


if __name__ == "__main__":
    test_validation_final_test_masks_train_and_validation_positives()
    test_fixed_epoch_mode_masks_training_positives_only()
    print("Evaluation masking audit: PASS")
