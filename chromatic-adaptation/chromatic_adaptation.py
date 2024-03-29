import numpy as np
import cv2

D65_WHITE_POINT_XYZ = np.array([0.95047, 1., 1.08883])

RGB_TO_XYZ = np.array([[0.412453, 0.357580, 0.180423], 
					   [0.212671, 0.715160, 0.072169], 
					   [0.019334, 0.119193, 0.950227]])

XYZ_TO_RGB = np.array([[3.240481, -1.537151, -0.498536],
					   [-0.969256, 1.875990, 0.0415560],
					   [0.055647, -0.204041, 1.057311]])

BRADFORD = np.array([[0.8951, 0.2664, -0.1614], 
					 [-0.7502, 1.7135, 0.0367], 
					 [0.0389, -0.0685, 1.0296]])

VON_KRIES = np.array([[0.40024, 0.70760, -0.08081],
					  [-0.22630, 1.16532, 0.04570],
					  [0.00000, 0.00000, 0.91822]])

SHARP = np.array([[1.2694, -0.0988, -0.1706],
				  [-0.8364, 1.8006, 0.0357],
				  [0.0297, -0.0315, 1.0018]])

CAT2000 = np.array([[0.7982, 0.3389, -0.1371],
					[-0.5918, 1.5512, 0.0406], 
					[0.0008, 0.2390, 0.9753]])

CAT02 = np.array([[0.7328, 0.4296, -0.1624],
				  [-0.7036, 1.6975, 0.0061],
				  [0.0030, 0.0136, 0.9834]])

def srgb_to_linear(srgb):
	# 'sRGB' in [0.0, 1.0]
	
	ln_rgb = srgb.copy()
	mask = ln_rgb > 0.04045
	ln_rgb[mask] = np.power((ln_rgb[mask] + 0.055) / 1.055, 2.4)
	ln_rgb[~mask] /= 12.92
	return ln_rgb

def linear_to_srgb(linear):
	# 'linear RGB' in [0.0, 1.0]
	
	srgb = linear.copy()
	mask = srgb > 0.0031308
	srgb[mask] = 1.055 * np.power(srgb[mask], 1 / 2.4) - 0.055
	srgb[~mask] *= 12.92
	return np.clip(srgb, 0.0, 1.0)

def srgb_to_xyz(srgb):
	# convert 'sRGB' to 'linear RGB'
	rgb = srgb_to_linear(srgb)
	# convert 'linear RGB' to 'XYZ'
	return rgb @ RGB_TO_XYZ.T

def xyz_to_srgb(xyz):
	# convert 'XYZ' to 'linear RGB'
	rgb = xyz @ XYZ_TO_RGB.T
	# convert back 'linear RGB' to 'sRGB'
	return linear_to_srgb(rgb)

def normalize_xyz(xyz):
	# normalize 'XYZ' with 'Y' so that 'Y' represents luminance
	return xyz / xyz[1]

def xyz_to_xy(xyz):
	return np.array([xyz[0] / xyz.sum(), xyz[1] / xyz.sum()])

def xy_to_xyz(xy, Y):
	x = xy[0]
	y = xy[1]
	return np.array([Y / y * x, Y, Y / (y*(1-x-y))])

def get_gray_world_illuminant(img):
	# image in sRGB with range [0.0, 1.0]
	# convert sRGB to linear RGB
	ln_img = srgb_to_linear(img)
	# mean of each channel
	avg_ch = ln_img.mean(axis=(0, 1))
	# convert back RGB mean values to sRGB
	return linear_to_srgb(avg_ch)

def get_colorchecker_coord(img):
	# initialize CCDetector object
	checker_detector = cv2.mcc.CCheckerDetector_create()
	# detect classic Macbeth 24 color grid chart
	has_chart = checker_detector.process(img, cv2.mcc.MCC24, 1)
	# if any chart present
	if has_chart:
		# ColorChecker chart coordinates
		# order - (tl, tr, br, bl)
		box = checker_detector.getListColorChecker()[0].getBox()
		min_x = int(min(box[0][0], box[3][0]))
		max_x = int(max(box[1][0], box[2][0]))
		min_y = int(min(box[0][1], box[1][1]))
		max_y = int(max(box[2][1], box[3][1]))
		coord = [(min_x, min_y), (max_x, max_y)]
		return [True, coord]
	else:
		return [False, []]

def get_cat_matrix(cat_type = 'BRADFORD'):
	if cat_type == 'BRADFORD':
		return BRADFORD
	elif cat_type == 'VON_KRIES':
		return VON_KRIES
	elif cat_type == 'SHARP':
		return SHARP
	elif cat_type == 'CAT2000':
		return CAT2000
	else:
		return CAT02
	
def xyz_to_lms(xyz, M):
	return xyz @ M.T

def get_gain(lms_src, lms_dst):
	return lms_dst / lms_src

def transform_lms(M, gain):
	return np.linalg.inv(M) @ np.diag(gain) @ M

def chromatic_adaptation_image(src_white_point, dst_white_point, src_img, cat_type = 'BRADFORD'):
	'''Convert image to destination illuminant conditions using chromatic adaptation

	Params:
	-------
		src_white_point: source illuminant white point in scaled [0, 1] sRGB values
		dst_white_point: destination illuminant white point in scaled [0, 1] sRGB values
		src_img: image to transform
		cat_type: chromatic adaptation transform type

	Returns:
	--------
		transformed_img: chromatic adaptation transformed image

	'''

	# convert white point in 'sRGB' to 'XYZ' 
	# and normalize 'XYZ' that 'Y' as luminance
	xyz_src = srgb_to_xyz(src_white_point)
	n_xyz_src = normalize_xyz(xyz_src)
	xyz_dst = srgb_to_xyz(dst_white_point)
	n_xyz_dst = normalize_xyz(xyz_dst)

	# get CAT type matrix
	cat_m = get_cat_matrix(cat_type)

	# convert 'XYZ' to 'LMS'
	lms_src = xyz_to_lms(n_xyz_src, cat_m)
	lms_dst = xyz_to_lms(n_xyz_dst, cat_m)
	# LMS gain by scaling destination with source LMS
	gain = get_gain(lms_src, lms_dst)

	# multiply CAT matrix with LMS gain factors
	ca_transform = transform_lms(cat_m, gain)
	
	# convert 'sRGB' source image to 'XYZ' 
	src_img_xyz = srgb_to_xyz(src_img)
	
	# apply CAT transform to image
	transformed_xyz = src_img_xyz @ ca_transform.T
	
	# convert back 'XYZ' to 'sRGB' image
	transformed_img = xyz_to_srgb(transformed_xyz)
	
	return transformed_img

def main(image, dst_white_point = [1.0, 1.0, 1.0], cat_type = 'BRADFORD'):
	'''Convert image to destination illuminant conditions using chromatic adaptation

	Params:
	-------
		image: source image to transform
		dst_white_point: a list of illuminant white point in scaled [0, 1] sRGB values
		cat_type: chromatic adaptation transform type

	Returns:
	--------
		ca_img: chromatic adaptation transformed image

	'''

	# image which generally in sRGB format
	src_img = image.copy()
	# reverse channel order from BGR to RGB and scale to 1.0
	r_img = src_img[:, :, ::-1] / 255

	# get source illuminant by illumination estimation
	src_white_point = np.array([1.0, 1.0, 1.0])
	has_chart, coord = get_colorchecker_coord(src_img)
	if has_chart:
		src_white_point = get_gray_world_illuminant(r_img[coord[0][1]: coord[1][1], coord[0][0]: coord[1][0]])
	else:
		src_white_point = get_gray_world_illuminant(r_img)
	dst_white_point = np.array(dst_white_point)

	# apply chromatic apatation for source image
	ca_img = chromatic_adaptation_image(src_white_point, dst_white_point, r_img, cat_type)

	# reverse channel order from RGB to BGR, and rescale to 255
	ca_img = (ca_img[:, :, ::-1] * 255).astype(np.uint8)

	return ca_img