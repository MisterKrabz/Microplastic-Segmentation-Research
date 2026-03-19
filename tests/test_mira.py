"""
Unit tests for the MIRA package.

Covers utils, metrics, drawing, and the runner classes (MiraYOLO, MiraSAM2,
MiraSequential) with mocked model backends so no GPU or weights are needed.

Run:  pytest tests/test_mira.py -v
"""

import os
import sys
import tempfile
import shutil
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mira.utils import (
    count_gt_instances,
    discover_yolo_versions,
    get_device,
    label_path_for_image,
    parse_yolo_seg_labels,
    resolve_yolo_model,
    scan_images,
)
from mira.metrics import (
    DatasetMetrics,
    ImageMetrics,
    compute_mask_iou,
    count_accuracy,
    match_masks,
)
from mira.drawing import draw_boxes, draw_masks, shrink_box


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture()
def tmp_dir():
    """Create a temporary directory tree and clean up after."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture()
def sample_image_dir(tmp_dir):
    """Populate tmp_dir with a fake dataset structure."""
    imgs = os.path.join(tmp_dir, "train", "images")
    lbls = os.path.join(tmp_dir, "train", "labels")
    os.makedirs(imgs)
    os.makedirs(lbls)

    for name in ("a.jpg", "b.png", "c.txt"):
        open(os.path.join(imgs, name), "w").close()

    with open(os.path.join(lbls, "a.txt"), "w") as f:
        f.write("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
        f.write("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

    with open(os.path.join(lbls, "b.txt"), "w") as f:
        f.write("")

    return tmp_dir


@pytest.fixture()
def models_dir(tmp_dir):
    """Create dummy .pt files matching the hunter-yolo naming convention."""
    for ver in ("0.2.3", "0.3.2", "0.4.4"):
        open(os.path.join(tmp_dir, f"hunter-yolo-v{ver}.pt"), "w").close()
    open(os.path.join(tmp_dir, "sam2.1_hiera_large.pt"), "w").close()
    return tmp_dir


def _make_bgr(h=100, w=100):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ======================================================================
# utils.py
# ======================================================================

class TestGetDevice:
    def test_explicit_device_returned(self):
        assert get_device("cpu") == "cpu"
        assert get_device("cuda:1") == "cuda:1"

    def test_none_returns_string(self):
        result = get_device(None)
        assert isinstance(result, str)
        assert result in ("cuda", "mps", "cpu")


class TestScanImages:
    def test_finds_images_in_directory(self, sample_image_dir):
        imgs_dir = os.path.join(sample_image_dir, "train", "images")
        result = scan_images(imgs_dir)
        basenames = [os.path.basename(p) for p in result]
        assert "a.jpg" in basenames
        assert "b.png" in basenames
        assert "c.txt" not in basenames

    def test_comma_separated_paths(self, sample_image_dir):
        imgs_dir = os.path.join(sample_image_dir, "train", "images")
        result = scan_images(f"{imgs_dir}, {imgs_dir}")
        assert len(result) == 2

    def test_single_file_path(self, sample_image_dir):
        fpath = os.path.join(sample_image_dir, "train", "images", "a.jpg")
        result = scan_images(fpath)
        assert len(result) == 1

    def test_nonexistent_returns_empty(self):
        result = scan_images("/no/such/path")
        assert result == []

    def test_non_image_file_ignored(self, sample_image_dir):
        txt = os.path.join(sample_image_dir, "train", "images", "c.txt")
        result = scan_images(txt)
        assert result == []


class TestLabelPathForImage:
    def test_images_to_labels_swap(self):
        path = os.path.join("data", "train", "images", "photo.jpg")
        expected = os.path.join("data", "train", "labels", "photo.txt")
        assert label_path_for_image(path) == expected

    def test_fallback_without_images_dir(self):
        path = os.path.join("folder", "photo.png")
        expected = os.path.join("folder", "photo.txt")
        assert label_path_for_image(path) == expected

    def test_nested_images_directory(self):
        path = os.path.join("a", "images", "sub", "img.tiff")
        result = label_path_for_image(path)
        assert "labels" in result
        assert result.endswith(".txt")


class TestParseYoloSegLabels:
    def test_parses_two_masks(self, sample_image_dir):
        lbl = os.path.join(sample_image_dir, "train", "labels", "a.txt")
        masks = parse_yolo_seg_labels(lbl, 100, 100)
        assert len(masks) == 2
        for m in masks:
            assert m.shape == (100, 100)
            assert m.dtype == np.uint8
            assert m.sum() > 0

    def test_missing_file_returns_empty(self):
        masks = parse_yolo_seg_labels("/no/file.txt", 100, 100)
        assert masks == []

    def test_empty_file_returns_empty(self, sample_image_dir):
        lbl = os.path.join(sample_image_dir, "train", "labels", "b.txt")
        masks = parse_yolo_seg_labels(lbl, 100, 100)
        assert masks == []

    def test_short_line_skipped(self, tmp_dir):
        lbl = os.path.join(tmp_dir, "short.txt")
        with open(lbl, "w") as f:
            f.write("0 0.5 0.5\n")
        masks = parse_yolo_seg_labels(lbl, 100, 100)
        assert masks == []


class TestCountGtInstances:
    def test_counts_nonempty_lines(self, sample_image_dir):
        lbl = os.path.join(sample_image_dir, "train", "labels", "a.txt")
        assert count_gt_instances(lbl) == 2

    def test_empty_file(self, sample_image_dir):
        lbl = os.path.join(sample_image_dir, "train", "labels", "b.txt")
        assert count_gt_instances(lbl) == 0

    def test_missing_file(self):
        assert count_gt_instances("/nope.txt") == 0


class TestDiscoverYoloVersions:
    def test_finds_all_versions(self, models_dir):
        versions = discover_yolo_versions(models_dir)
        assert set(versions.keys()) == {"0.2.3", "0.3.2", "0.4.4"}
        for v, path in versions.items():
            assert path.endswith(f"hunter-yolo-v{v}.pt")

    def test_ignores_non_matching_files(self, models_dir):
        versions = discover_yolo_versions(models_dir)
        assert "sam2" not in str(versions)

    def test_nonexistent_dir(self):
        assert discover_yolo_versions("/no/such/dir") == {}


class TestResolveYoloModel:
    def test_explicit_path(self, models_dir):
        path = os.path.join(models_dir, "hunter-yolo-v0.4.4.pt")
        assert resolve_yolo_model(path) == path

    def test_version_lookup(self, models_dir):
        path = resolve_yolo_model(version="0.4.4", models_dir=models_dir)
        assert path.endswith("hunter-yolo-v0.4.4.pt")

    def test_version_with_v_prefix(self, models_dir):
        path = resolve_yolo_model(version="v0.3.2", models_dir=models_dir)
        assert path.endswith("hunter-yolo-v0.3.2.pt")

    def test_missing_version_raises(self, models_dir):
        with pytest.raises(FileNotFoundError, match="v9.9.9"):
            resolve_yolo_model(version="9.9.9", models_dir=models_dir)

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_yolo_model("/no/such/model.pt")

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            resolve_yolo_model()


# ======================================================================
# metrics.py
# ======================================================================

class TestComputeMaskIou:
    def test_identical_masks(self):
        m = np.ones((50, 50), dtype=np.uint8)
        assert compute_mask_iou(m, m) == pytest.approx(1.0)

    def test_disjoint_masks(self):
        m1 = np.zeros((50, 50), dtype=np.uint8)
        m2 = np.zeros((50, 50), dtype=np.uint8)
        m1[:25, :] = 1
        m2[25:, :] = 1
        assert compute_mask_iou(m1, m2) == pytest.approx(0.0)

    def test_half_overlap(self):
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m1[:, :60] = 1
        m2[:, 40:] = 1
        intersection = 100 * 20
        union = 100 * 60 + 100 * 60 - intersection
        expected = intersection / union
        assert compute_mask_iou(m1, m2) == pytest.approx(expected, abs=1e-5)

    def test_both_empty(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        assert compute_mask_iou(m, m) == 0.0


class TestMatchMasks:
    def test_both_empty(self):
        mean_iou, tp, fp, fn, ious = match_masks([], [])
        assert mean_iou == 1.0
        assert tp == fp == fn == 0

    def test_no_gt(self):
        pred = [np.ones((10, 10), dtype=np.uint8)]
        mean_iou, tp, fp, fn, ious = match_masks(pred, [])
        assert mean_iou == 0.0
        assert fp == 1

    def test_no_preds(self):
        gt = [np.ones((10, 10), dtype=np.uint8)]
        mean_iou, tp, fp, fn, ious = match_masks([], gt)
        assert mean_iou == 0.0
        assert fn == 1

    def test_perfect_match(self):
        m = np.ones((10, 10), dtype=np.uint8)
        mean_iou, tp, fp, fn, ious = match_masks([m], [m], iou_threshold=0.5)
        assert mean_iou == pytest.approx(1.0)
        assert tp == 1
        assert fp == 0
        assert fn == 0

    def test_mixed_match(self):
        m1 = np.zeros((50, 50), dtype=np.uint8)
        m1[:25, :] = 1
        m2 = np.zeros((50, 50), dtype=np.uint8)
        m2[25:, :] = 1

        mean_iou, tp, fp, fn, ious = match_masks([m1, m2], [m1], iou_threshold=0.5)
        assert tp == 1
        assert fp == 1
        assert fn == 0


class TestCountAccuracy:
    def test_exact_match(self):
        assert count_accuracy(5, 5) == 1.0

    def test_zero_gt_zero_pred(self):
        assert count_accuracy(0, 0) == 1.0

    def test_zero_gt_nonzero_pred(self):
        assert count_accuracy(0, 3) == 0.0

    def test_overcount(self):
        assert count_accuracy(10, 15) == pytest.approx(0.5)

    def test_undercount(self):
        assert count_accuracy(10, 5) == pytest.approx(0.5)

    def test_clamp_to_zero(self):
        assert count_accuracy(2, 100) == 0.0


class TestDatasetMetrics:
    def _make_img_metrics(self, pred, gt, iou, tp, fp, fn):
        return ImageMetrics("img.jpg", pred, gt, iou, tp, fp, fn)

    def test_empty(self):
        dm = DatasetMetrics()
        assert dm.n_images == 0
        assert dm.mean_iou == 0.0
        assert dm.precision == 0.0
        assert dm.recall == 0.0
        assert dm.f1 == 0.0

    def test_aggregation(self):
        dm = DatasetMetrics(images=[
            self._make_img_metrics(3, 3, 0.9, 3, 0, 0),
            self._make_img_metrics(2, 3, 0.6, 2, 0, 1),
        ])
        assert dm.n_images == 2
        assert dm.total_gt == 6
        assert dm.total_pred == 5
        assert dm.total_tp == 5
        assert dm.total_fp == 0
        assert dm.total_fn == 1
        assert dm.mean_iou == pytest.approx(0.75)
        assert dm.precision == pytest.approx(1.0)
        assert dm.recall == pytest.approx(5 / 6)

    def test_f1(self):
        dm = DatasetMetrics(images=[
            self._make_img_metrics(4, 4, 0.8, 3, 1, 1),
        ])
        p = 3 / 4
        r = 3 / 4
        expected_f1 = 2 * p * r / (p + r)
        assert dm.f1 == pytest.approx(expected_f1)

    def test_summary_is_string(self):
        dm = DatasetMetrics(images=[
            self._make_img_metrics(2, 2, 0.7, 2, 0, 0),
        ])
        s = dm.summary()
        assert isinstance(s, str)
        assert "Mean IoU" in s


# ======================================================================
# drawing.py
# ======================================================================

class TestDrawBoxes:
    def test_no_boxes_returns_copy(self):
        img = _make_bgr()
        out = draw_boxes(img, np.empty((0, 4)))
        assert out.shape == img.shape
        np.testing.assert_array_equal(out, img)

    def test_none_boxes_returns_copy(self):
        img = _make_bgr()
        out = draw_boxes(img, None)
        np.testing.assert_array_equal(out, img)

    def test_single_box_modifies_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        out = draw_boxes(img, boxes, show_labels=False)
        assert not np.array_equal(out, img)

    def test_with_labels(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.array([[10, 20, 80, 80]], dtype=np.float32)
        confs = np.array([0.95])
        names = ["particle"]
        out = draw_boxes(img, boxes, confs, names, show_labels=True, show_class_name=True)
        assert not np.array_equal(out, img)

    def test_does_not_mutate_input(self):
        img = _make_bgr()
        original = img.copy()
        boxes = np.array([[5, 5, 50, 50]], dtype=np.float32)
        draw_boxes(img, boxes)
        np.testing.assert_array_equal(img, original)


class TestDrawMasks:
    def test_empty_masks(self):
        img = _make_bgr()
        out = draw_masks(img, [])
        assert out.shape == img.shape

    def test_with_valid_mask(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0
        out = draw_masks(img, [mask])
        assert not np.array_equal(out, img)

    def test_none_mask_skipped(self):
        img = _make_bgr()
        out = draw_masks(img, [None])
        assert out.shape == img.shape

    def test_zero_mask_skipped(self):
        img = _make_bgr()
        mask = np.zeros((100, 100), dtype=np.float32)
        out = draw_masks(img, [mask])
        assert out.shape == img.shape


class TestShrinkBox:
    def test_zero_frac_unchanged(self):
        box = np.array([10, 10, 90, 90], dtype=np.float32)
        result = shrink_box(box, 100, 100, 0.0)
        np.testing.assert_array_equal(result, box)

    def test_shrinks_inward(self):
        box = np.array([0, 0, 100, 100], dtype=np.float32)
        result = shrink_box(box, 200, 200, 0.1)
        assert result[0] > box[0]
        assert result[1] > box[1]
        assert result[2] < box[2]
        assert result[3] < box[3]

    def test_clamps_to_image_bounds(self):
        box = np.array([0, 0, 10, 10], dtype=np.float32)
        result = shrink_box(box, 50, 50, 0.1)
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] <= 49
        assert result[3] <= 49

    def test_too_much_shrink_reverts(self):
        box = np.array([10, 10, 12, 12], dtype=np.float32)
        result = shrink_box(box, 100, 100, 0.9)
        assert result[2] > result[0]
        assert result[3] > result[1]


# ======================================================================
# yolo.py  (mocked model)
# ======================================================================

class TestYOLOResult:
    def test_count_property(self):
        from mira.yolo import YOLOResult
        r = YOLOResult(
            image_path="test.jpg",
            image_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
            boxes_xyxy=np.array([[0, 0, 5, 5], [1, 1, 6, 6]], dtype=np.float32),
            confidences=np.array([0.9, 0.8]),
            class_ids=np.array([0, 0]),
            class_names=["a", "a"],
        )
        assert r.count == 2

    def test_empty_result(self):
        from mira.yolo import YOLOResult
        r = YOLOResult(
            image_path="empty.jpg",
            image_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=int),
        )
        assert r.count == 0


class TestMiraYOLO:
    def _mock_boxes(self, n=2):
        """Build a mock ultralytics Boxes object."""
        import torch

        boxes_mock = mock.MagicMock()
        boxes_mock.__len__ = mock.MagicMock(return_value=n)
        boxes_mock.__bool__ = mock.MagicMock(return_value=n > 0)
        boxes_mock.xyxy = torch.tensor(
            [[10, 10, 50, 50], [60, 60, 90, 90]][:n], dtype=torch.float32
        )
        boxes_mock.conf = torch.tensor([0.95, 0.80][:n], dtype=torch.float32)
        boxes_mock.cls = torch.tensor([0, 0][:n], dtype=torch.float32)
        return boxes_mock

    @mock.patch("mira.yolo.YOLO")
    def test_predict_from_path(self, MockYOLO, tmp_dir):
        import cv2

        img_path = os.path.join(tmp_dir, "test.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        r0 = mock.MagicMock()
        r0.boxes = self._mock_boxes(2)
        r0.names = {0: "particle"}
        MockYOLO.return_value.predict.return_value = [r0]

        from mira.yolo import MiraYOLO

        yolo = MiraYOLO.__new__(MiraYOLO)
        yolo.device = "cpu"
        yolo.conf = 0.3
        yolo.iou = 0.25
        yolo.imgsz = 640
        yolo.model = MockYOLO.return_value

        result = yolo.predict(img_path)
        assert result.count == 2
        assert result.image_path == img_path
        assert result.class_names == ["particle", "particle"]
        assert result.image_rgb.shape == (100, 100, 3)

    @mock.patch("mira.yolo.YOLO")
    def test_predict_from_ndarray(self, MockYOLO):
        r0 = mock.MagicMock()
        r0.boxes = self._mock_boxes(1)
        r0.names = {0: "mp"}
        MockYOLO.return_value.predict.return_value = [r0]

        from mira.yolo import MiraYOLO

        yolo = MiraYOLO.__new__(MiraYOLO)
        yolo.device = "cpu"
        yolo.conf = 0.3
        yolo.iou = 0.25
        yolo.imgsz = 640
        yolo.model = MockYOLO.return_value

        img = np.zeros((80, 80, 3), dtype=np.uint8)
        result = yolo.predict(img)
        assert result.count == 1
        assert result.image_path == "<ndarray>"

    @mock.patch("mira.yolo.YOLO")
    def test_predict_no_detections(self, MockYOLO):
        r0 = mock.MagicMock()
        r0.boxes = self._mock_boxes(0)
        r0.names = {}
        MockYOLO.return_value.predict.return_value = [r0]

        from mira.yolo import MiraYOLO

        yolo = MiraYOLO.__new__(MiraYOLO)
        yolo.device = "cpu"
        yolo.conf = 0.3
        yolo.iou = 0.25
        yolo.imgsz = 640
        yolo.model = MockYOLO.return_value

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = yolo.predict(img)
        assert result.count == 0
        assert len(result.class_names) == 0

    @mock.patch("mira.yolo.YOLO")
    def test_predict_batch(self, MockYOLO, tmp_dir):
        import cv2

        paths = []
        for name in ("a.jpg", "b.jpg"):
            p = os.path.join(tmp_dir, name)
            cv2.imwrite(p, np.zeros((50, 50, 3), dtype=np.uint8))
            paths.append(p)

        r0 = mock.MagicMock()
        r0.boxes = self._mock_boxes(1)
        r0.names = {0: "p"}
        MockYOLO.return_value.predict.return_value = [r0]

        from mira.yolo import MiraYOLO

        yolo = MiraYOLO.__new__(MiraYOLO)
        yolo.device = "cpu"
        yolo.conf = 0.3
        yolo.iou = 0.25
        yolo.imgsz = 640
        yolo.model = MockYOLO.return_value

        results = yolo.predict_batch(paths)
        assert len(results) == 2

    def test_available_versions(self, models_dir):
        from mira.yolo import MiraYOLO

        versions = MiraYOLO.available_versions(models_dir)
        assert versions == ["0.2.3", "0.3.2", "0.4.4"]


# ======================================================================
# sam2.py  (mocked model)
# ======================================================================

class TestSAM2Result:
    def test_count_property(self):
        from mira.sam2 import SAM2Result
        m1 = np.ones((10, 10), dtype=np.uint8)
        r = SAM2Result(
            image_path="x.jpg",
            image_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
            masks=[m1, m1],
        )
        assert r.count == 2


class TestMiraSAM2:
    def test_predict_produces_masks(self):
        from mira.sam2 import MiraSAM2

        mask_out = np.ones((1, 80, 80), dtype=np.float32)
        predictor_inst = mock.MagicMock()
        predictor_inst.predict.return_value = (
            mask_out[np.newaxis, ...],
            np.array([0.99]),
            None,
        )

        sam = MiraSAM2.__new__(MiraSAM2)
        sam.device = "cpu"
        sam.box_shrink = 0.1
        sam.predictor = predictor_inst

        img = np.zeros((80, 80, 3), dtype=np.uint8)
        boxes = [[10, 10, 70, 70]]
        result = sam.predict(img, boxes)

        assert result.count == 1
        assert result.masks[0].shape == (80, 80)

    def test_predict_empty_boxes(self):
        from mira.sam2 import MiraSAM2

        sam = MiraSAM2.__new__(MiraSAM2)
        sam.device = "cpu"
        sam.box_shrink = 0.1
        sam.predictor = mock.MagicMock()

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = sam.predict(img, np.empty((0, 4)))
        assert result.count == 0


# ======================================================================
# sequential.py  (mocked models)
# ======================================================================

class TestSequentialResult:
    def test_pred_count(self):
        from mira.sequential import SequentialResult
        r = SequentialResult(
            image_path="t.jpg",
            image_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
            boxes_xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
            confidences=np.array([0.9]),
            class_ids=np.array([0]),
            class_names=["p"],
            masks=[np.ones((10, 10), dtype=np.uint8)],
        )
        assert r.pred_count == 1


class TestMiraSequential:
    def _build_mocked_pipeline(self, n_boxes=1):
        import torch

        boxes_mock = mock.MagicMock()
        boxes_mock.__len__ = mock.MagicMock(return_value=n_boxes)
        boxes_mock.__bool__ = mock.MagicMock(return_value=n_boxes > 0)
        boxes_mock.xyxy = torch.tensor(
            [[10, 10, 50, 50]][:n_boxes], dtype=torch.float32
        )
        boxes_mock.conf = torch.tensor([0.9][:n_boxes], dtype=torch.float32)
        boxes_mock.cls = torch.tensor([0][:n_boxes], dtype=torch.float32)

        r0 = mock.MagicMock()
        r0.boxes = boxes_mock
        r0.names = {0: "particle"}

        yolo_mock = mock.MagicMock()
        yolo_mock.predict.return_value = [r0]

        predictor_mock = mock.MagicMock()
        mask_out = np.ones((1, 80, 80), dtype=np.float32)
        predictor_mock.predict.return_value = (
            mask_out[np.newaxis, ...],
            np.array([0.99]),
            None,
        )

        from mira.sequential import MiraSequential

        pipeline = MiraSequential.__new__(MiraSequential)
        pipeline.device = "cpu"
        pipeline.conf = 0.3
        pipeline.iou = 0.25
        pipeline.imgsz = 640
        pipeline.box_shrink = 0.1
        pipeline.yolo = yolo_mock
        pipeline.predictor = predictor_mock
        return pipeline

    def test_predict_ndarray(self):
        pipeline = self._build_mocked_pipeline(n_boxes=1)
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        result = pipeline.predict(img)
        assert result.pred_count == 1
        assert len(result.masks) == 1

    def test_predict_no_detections(self):
        pipeline = self._build_mocked_pipeline(n_boxes=0)
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        result = pipeline.predict(img)
        assert result.pred_count == 0
        assert result.masks == []

    def test_predict_batch(self, tmp_dir):
        import cv2

        pipeline = self._build_mocked_pipeline(n_boxes=1)
        paths = []
        for name in ("img1.jpg", "img2.jpg"):
            p = os.path.join(tmp_dir, name)
            cv2.imwrite(p, np.zeros((80, 80, 3), dtype=np.uint8))
            paths.append(p)

        results = pipeline.predict_batch(paths)
        assert len(results) == 2
        assert all(r.pred_count == 1 for r in results)


# ======================================================================
# __init__.py  (public API surface)
# ======================================================================

class TestPublicAPI:
    def test_version(self):
        from mira import __version__
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_all_exports_importable(self):
        import mira
        for name in mira.__all__:
            assert hasattr(mira, name), f"mira.{name} not found"
