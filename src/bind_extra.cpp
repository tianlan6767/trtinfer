#include "bind_extra.h"

#include "pybind_cvmat.hpp"

#include "cv_calib/calib.h"
#include "cv_code/code.h"
#include "cv_feature/feature.h"
#include "cv_seg/seg.h"
#include "cv_sim/similarity.h"

#include <pybind11/stl.h>

#if __has_include(<opencv2/aruco.hpp>)
#include <opencv2/aruco.hpp>
#endif

namespace py = pybind11;

void bindExtra(py::module_& m)
{
	py::class_<sim::SimilarityScore>(m, "SimilarityScore")
		.def(py::init<>())
		.def_readwrite("found", &sim::SimilarityScore::found)
		.def_readwrite("ncc", &sim::SimilarityScore::ncc)
		.def_readwrite("zncc", &sim::SimilarityScore::zncc)
		.def_readwrite("histCorr", &sim::SimilarityScore::histCorr)
		.def_readwrite("histBhattacharyya", &sim::SimilarityScore::histBhattacharyya)
		.def_readwrite("ssim", &sim::SimilarityScore::ssim);

	py::class_<sim::LocateResult>(m, "LocateResult")
		.def(py::init<>())
		.def_readwrite("found", &sim::LocateResult::found)
		.def_readwrite("score", &sim::LocateResult::score)
		.def_readwrite("topLeft", &sim::LocateResult::topLeft)
		.def_readwrite("center", &sim::LocateResult::center)
		.def_readwrite("width", &sim::LocateResult::width)
		.def_readwrite("height", &sim::LocateResult::height);

	m.def("compare_similarity", &sim::compare, py::arg("a"), py::arg("b"));
	m.def("locate_zncc", &sim::locateZncc, py::arg("image"), py::arg("templ"), py::arg("threshold") = 0.6);
	m.def("locate_ncc", &sim::locateNcc, py::arg("image"), py::arg("templ"), py::arg("threshold") = 0.6);

	py::class_<seg::Contour>(m, "Contour")
		.def(py::init<>())
		.def_readwrite("area", &seg::Contour::area)
		.def_readwrite("perimeter", &seg::Contour::perimeter)
		.def_readwrite("center", &seg::Contour::center)
		.def_readwrite("left", &seg::Contour::left)
		.def_readwrite("top", &seg::Contour::top)
		.def_readwrite("width", &seg::Contour::width)
		.def_readwrite("height", &seg::Contour::height)
		.def_readwrite("points", &seg::Contour::points);

	py::class_<seg::Region>(m, "Region")
		.def(py::init<>())
		.def_readwrite("label", &seg::Region::label)
		.def_readwrite("area", &seg::Region::area)
		.def_readwrite("center", &seg::Region::center)
		.def_readwrite("left", &seg::Region::left)
		.def_readwrite("top", &seg::Region::top)
		.def_readwrite("width", &seg::Region::width)
		.def_readwrite("height", &seg::Region::height);

	py::class_<seg::WatershedResult>(m, "WatershedResult")
		.def(py::init<>())
		.def_readwrite("found", &seg::WatershedResult::found)
		.def_readwrite("labels", &seg::WatershedResult::labels)
		.def_readwrite("regions", &seg::WatershedResult::regions);

	m.def("distance_transform", &seg::distanceTransform, py::arg("binary"));
	m.def("canny", &seg::canny, py::arg("image"), py::arg("low") = 50.0, py::arg("high") = 150.0, py::arg("aperture") = 3);
	m.def("extract_contours", &seg::extractContours, py::arg("binary"), py::arg("external_only") = true);
	m.def("watershed", &seg::watershed, py::arg("image"), py::arg("markers"));
	m.def("split_blobs", &seg::splitBlobs, py::arg("binary"), py::arg("min_distance") = 8.0, py::arg("dist_ratio") = 0.4);

	py::enum_<feature::Detector>(m, "FeatureDetector")
		.value("ORB", feature::Detector::ORB)
		.value("SIFT", feature::Detector::SIFT)
		.export_values();

	py::class_<feature::MatchResult>(m, "FeatureMatch")
		.def(py::init<>())
		.def_readwrite("found", &feature::MatchResult::found)
		.def_readwrite("method", &feature::MatchResult::method)
		.def_readwrite("keypointsModel", &feature::MatchResult::keypointsModel)
		.def_readwrite("keypointsScene", &feature::MatchResult::keypointsScene)
		.def_readwrite("matches", &feature::MatchResult::matches)
		.def_readwrite("inliers", &feature::MatchResult::inliers)
		.def_readwrite("homography", &feature::MatchResult::homography)
		.def_readwrite("modelPoints", &feature::MatchResult::modelPoints)
		.def_readwrite("scenePoints", &feature::MatchResult::scenePoints)
		.def_readwrite("inlierMask", &feature::MatchResult::inlierMask)
		.def_readwrite("corners", &feature::MatchResult::corners);

	m.def("sift_available", &feature::siftAvailable);
	m.def("match_features", &feature::matchHomography,
		  py::arg("scene"), py::arg("model"),
		  py::arg("detector") = feature::Detector::ORB,
		  py::arg("ratio") = 0.75, py::arg("ransac_reproj") = 3.0);

	py::class_<code::DecodeHit>(m, "DecodeHit")
		.def(py::init<>())
		.def_readwrite("found", &code::DecodeHit::found)
		.def_readwrite("text", &code::DecodeHit::text)
		.def_readwrite("format", &code::DecodeHit::format)
		.def_readwrite("polygon", &code::DecodeHit::polygon);

	py::class_<code::OcrResult>(m, "OcrResult")
		.def(py::init<>())
		.def_readwrite("found", &code::OcrResult::found)
		.def_readwrite("text", &code::OcrResult::text)
		.def_readwrite("backend", &code::OcrResult::backend);

	m.def("barcode_available", &code::barcodeAvailable);
	m.def("ocr_available", &code::ocrAvailable);
	m.def("decode_qr", &code::decodeQr, py::arg("image"));
	m.def("decode_barcode", &code::decodeBarcode, py::arg("image"));
	m.def("decode_codes", &code::decodeCodes, py::arg("image"));
	m.def("recognize_text", &code::recognizeText, py::arg("image"), py::arg("lang") = std::string("eng"));

	py::enum_<calib::BoardType>(m, "BoardType")
		.value("Chessboard", calib::BoardType::Chessboard)
		.value("Charuco", calib::BoardType::Charuco)
		.value("CirclesSymmetric", calib::BoardType::CirclesSymmetric)
		.value("CirclesAsymmetric", calib::BoardType::CirclesAsymmetric)
		.value("ArucoGrid", calib::BoardType::ArucoGrid)
		.value("CharucoDiamond", calib::BoardType::CharucoDiamond)
		.value("AprilTag", calib::BoardType::AprilTag)
		.export_values();

	py::enum_<calib::HandEyeMethod>(m, "HandEyeMethod")
		.value("Tsai", calib::HandEyeMethod::Tsai)
		.value("Park", calib::HandEyeMethod::Park)
		.value("Horaud", calib::HandEyeMethod::Horaud)
		.value("Andreff", calib::HandEyeMethod::Andreff)
		.value("Daniilidis", calib::HandEyeMethod::Daniilidis)
		.export_values();

	py::class_<calib::BoardConfig>(m, "BoardConfig")
		.def(py::init<>())
		.def_readwrite("type", &calib::BoardConfig::type)
		.def_readwrite("squaresX", &calib::BoardConfig::squaresX)
		.def_readwrite("squaresY", &calib::BoardConfig::squaresY)
		.def_readwrite("squareLength", &calib::BoardConfig::squareLength)
		.def_readwrite("markerLength", &calib::BoardConfig::markerLength)
		.def_readwrite("markerSeparation", &calib::BoardConfig::markerSeparation)
		.def_readwrite("dictionary", &calib::BoardConfig::dictionary)
		.def_readwrite("firstMarker", &calib::BoardConfig::firstMarker)
		.def_readwrite("useSb", &calib::BoardConfig::useSb)
		.def_readwrite("subpixWin", &calib::BoardConfig::subpixWin)
		.def_readwrite("subpixIters", &calib::BoardConfig::subpixIters)
		.def_readwrite("subpixEps", &calib::BoardConfig::subpixEps);

	py::class_<calib::BoardDetection>(m, "BoardDetection")
		.def(py::init<>())
		.def_readwrite("found", &calib::BoardDetection::found)
		.def_readwrite("subpixel", &calib::BoardDetection::subpixel)
		.def_readwrite("board", &calib::BoardDetection::board)
		.def_readwrite("refine", &calib::BoardDetection::refine)
		.def_readwrite("count", &calib::BoardDetection::count)
		.def_readwrite("imagePoints", &calib::BoardDetection::imagePoints)
		.def_readwrite("objectPoints", &calib::BoardDetection::objectPoints)
		.def_readwrite("ids", &calib::BoardDetection::ids);

	py::class_<calib::CalibResult>(m, "CalibResult")
		.def(py::init<>())
		.def_readwrite("found", &calib::CalibResult::found)
		.def_readwrite("rms", &calib::CalibResult::rms)
		.def_readwrite("width", &calib::CalibResult::width)
		.def_readwrite("height", &calib::CalibResult::height)
		.def_readwrite("cameraMatrix", &calib::CalibResult::cameraMatrix)
		.def_readwrite("distCoeffs", &calib::CalibResult::distCoeffs)
		.def_readwrite("rvecs", &calib::CalibResult::rvecs)
		.def_readwrite("tvecs", &calib::CalibResult::tvecs);

	py::class_<calib::HandEyeResult>(m, "HandEyeResult")
		.def(py::init<>())
		.def_readwrite("found", &calib::HandEyeResult::found)
		.def_readwrite("R", &calib::HandEyeResult::R)
		.def_readwrite("t", &calib::HandEyeResult::t);

	m.def("aruco_available", &calib::arucoAvailable);
	m.def("draw_board", &calib::drawBoard, py::arg("config"), py::arg("pixels_per_unit") = 40, py::arg("margin") = 30);
	m.def("detect_board", &calib::detectBoard, py::arg("image"), py::arg("config") = calib::BoardConfig());
	m.def("calibrate_detections", &calib::calibrateDetections, py::arg("detections"), py::arg("width"), py::arg("height"));
	m.def("calibrate_images", &calib::calibrateImages, py::arg("images"), py::arg("config") = calib::BoardConfig());
	m.def("undistort", &calib::undistortImage, py::arg("image"), py::arg("camera_matrix"), py::arg("dist_coeffs"));
	m.def("save_calibration", &calib::saveCalibration, py::arg("path"), py::arg("result"));
	m.def("load_calibration", &calib::loadCalibration, py::arg("path"));
	m.def("calibrate_hand_eye", &calib::calibrateHandEye,
		  py::arg("R_gripper2base"), py::arg("t_gripper2base"),
		  py::arg("R_target2cam"), py::arg("t_target2cam"),
		  py::arg("method") = calib::HandEyeMethod::Tsai);

#if __has_include(<opencv2/aruco.hpp>)
	m.attr("DICT_4X4_50") = static_cast<int>(cv::aruco::DICT_4X4_50);
	m.attr("DICT_5X5_100") = static_cast<int>(cv::aruco::DICT_5X5_100);
	m.attr("DICT_APRILTAG_16h5") = static_cast<int>(cv::aruco::DICT_APRILTAG_16h5);
	m.attr("DICT_APRILTAG_25h9") = static_cast<int>(cv::aruco::DICT_APRILTAG_25h9);
	m.attr("DICT_APRILTAG_36h10") = static_cast<int>(cv::aruco::DICT_APRILTAG_36h10);
	m.attr("DICT_APRILTAG_36h11") = static_cast<int>(cv::aruco::DICT_APRILTAG_36h11);
#else
	m.attr("DICT_4X4_50") = 0;
	m.attr("DICT_5X5_100") = 4;
	m.attr("DICT_APRILTAG_16h5") = 17;
	m.attr("DICT_APRILTAG_25h9") = 18;
	m.attr("DICT_APRILTAG_36h10") = 19;
	m.attr("DICT_APRILTAG_36h11") = 20;
#endif
}
