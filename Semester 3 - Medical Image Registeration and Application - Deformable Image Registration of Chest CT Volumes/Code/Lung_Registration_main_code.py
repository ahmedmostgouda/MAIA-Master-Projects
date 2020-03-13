##LUNG REGISTRATION BY PATRICIA, AHMED AND ZAKIA

##EXECUTE THIS CODE AND THE RESULTS WOULD BE AUTOMATICALLY SAVED IN /EXPERIMENTS FOLDER

import sys  
import os
import numpy as np
import glob
import nibabel as nib
from xlwt import Workbook
from xlrd import open_workbook
from xlutils.copy import copy


#configuration of the variables
img_example = 'copd3'
registration = True #TO COMPUTE THE REGISTRATION
registration_type = ["Noisy","Noiseless"]
registration_type = registration_type[0]
transformix = True #TO COMPUTE THE TRANSFORMIX
experiment_type = ["GivenImage","NoNegativeValues","ImageROI"]
experiment_type = experiment_type[2]

#configuration of the paths
options = {}
options["train"] = 'train_img/'
options["parameters"] = "parameters/"

if registration_type == "Noisy":
  options["affine"] = options["parameters"]+"Par0000affine.txt"
  options["bspline"] = options["parameters"]+"Par0000bspline_noisy.txt"

if registration_type == "Noiseless":
  options["affine"] = options["parameters"]+"Par0000affine.txt"
  options["bspline"] = options["parameters"]+"Par0000bspline_noiseless.txt"

options["points"] = "output_points/"

options["points_modif"] = "points_modif/"+ img_example+"_300_iBH_xyz_r1_elastix.txt"
options["experiments"] = "experiments/"
options["registration"] = options["experiments"]+experiment_type+"/"+img_example+"/"+registration_type
options["result_elastix"] = options["registration"]+"/"+"TransformParameters.1.txt"
options["transformix"] = options["registration"]+"/"+options["points"]
options["excel"] = options["experiments"] +experiment_type+"_"+registration_type

if not os.path.isdir(options["experiments"]+experiment_type):
        os.mkdir(options["experiments"]+experiment_type)

if not os.path.isdir(options["experiments"]+experiment_type+"/"+img_example):
        os.mkdir(options["experiments"]+experiment_type+"/"+img_example)

if not os.path.isdir(options["registration"]):
        os.mkdir(options["registration"])

if not os.path.isdir(options["transformix"]):
        os.mkdir(options["transformix"])


#load the images and the landmark points
def load_img_points(name_img):
  training_scans = sorted(os.listdir(options['train']))

  if experiment_type == "GivenImage":
    input_train_fixed = {scan: [os.path.join(options['train'], scan, scan + '_iBHCT.nii')]  for scan in training_scans}
    input_train_moved = {scan: [os.path.join(options['train'], scan, scan + '_eBHCT.nii')]  for scan in training_scans}


  if experiment_type == "NoNegativeValues":
    input_train_fixed = {scan: [os.path.join(options['train'], scan,"pos_"+ scan + '_iBHCT.nii')]  for scan in training_scans}
    input_train_moved = {scan: [os.path.join(options['train'], scan,"pos_"+ scan + '_eBHCT.nii')]  for scan in training_scans}


  if experiment_type == "ImageROI":
    input_train_fixed = {scan: [os.path.join(options['train'], scan, "seg_"+scan + '_iBHCT.nii')]  for scan in training_scans}
    input_train_moved = {scan: [os.path.join(options['train'], scan, "seg_"+scan + '_eBHCT.nii')]  for scan in training_scans}


  input_point_fixed = {scan: [os.path.join(options['train'], scan, scan + '_300_iBH_xyz_r1.txt')]  for scan in training_scans}
  input_point_moved = {scan: [os.path.join(options['train'], scan, scan + '_300_eBH_xyz_r1.txt')]  for scan in training_scans}

  for k in input_point_fixed:
    print(k)

    if name_img in k:
      image_fixed_data = nib.load(str(input_train_fixed[name_img])[2:-2])
      image_moved_data = nib.load(str(input_train_moved[name_img])[2:-2])
      pixel_dim = image_fixed_data.header['pixdim'][1:4]
      points_fixed = np.loadtxt(str(input_point_fixed[name_img])[2:-2])*pixel_dim
      points_moved = np.loadtxt(str(input_point_moved[name_img])[2:-2])*pixel_dim
             
  
  return str(input_train_fixed[name_img])[2:-2], str(input_train_moved[name_img])[2:-2], str(input_point_fixed[name_img])[2:-2], str(input_point_moved[name_img])[2:-2], points_fixed,points_moved,pixel_dim


def registration_errors(reference_fixed_point_list, reference_moving_point_list):
  """
  Distances between points transformed by the given transformation and their
  location in another coordinate system. When the points are only used to 
  evaluate registration accuracy (not used in the registration) this is the 
  Target Registration Error (TRE).
  
  Args:
      reference_fixed_point_list (list(tuple-like)): Points in fixed image 
                                                     cooredinate system.
      reference_moving_point_list (list(tuple-like)): Points in moving image 
                                                      cooredinate system.

      min_err, max_err (float): color range is linearly stretched between min_err 
                                and max_err. If these values are not given then
                                the range of errors computed from the data is used.

  Returns:
   (mean, std, min, max) (float, float, float, float, [float]): 
    TRE statistics and original TREs.
  """

  errors = [np.linalg.norm(np.array(reference_fixed_point_list) -  np.array(reference_moving_point_list),axis = 1)]
  min_errors = np.min(errors)
  max_errors = np.max(errors)


  return (np.mean(errors), np.std(errors), min_errors, max_errors, errors) 


def transformix2np(path_transformix, no_points=300):
    """
    Reads and transforms the transformix output to ndarray to use in TRE function
    
    Parameters:    
        path_transformix (string): path to transformix output points txt file
        no_points(int): number of points in transformix file
    
    Returns:
        landmarks_array (ndarray): transformed points 
    """
    import re
    landmarks = open(path_transformix, "r")
    reg_expr = r'OutputIndexFixed = \[([\d.\s\-]+)\]'
    landmarks_array = np.zeros((no_points, 3))

    for index, line in enumerate(landmarks):
        match_obj = re.search(reg_expr, line, re.M)
        coords = match_obj.group(1).split()
        coords = [round(float(c)) for c in coords]
        landmarks_array[index,:] = coords
    return landmarks_array


image_fixed_data, image_moved_data,path_points_fixed, path_points_moved, points_fixed, points_moved,pixel_dim = load_img_points(img_example)

#execute elastix command to do the registration
if registration == True:
  command_elastix = "elastix -f "+image_fixed_data+" -m " + \
      image_moved_data+" -out "+options["registration"] + \
      " -p "+options["affine"]+" -p "+options["bspline"]
  os.system(command_elastix)


#execute transformix command to do the transformation of the points
if transformix == True:
  command_transformix = "transformix -def "+ options["points_modif"] + " -out " +options["transformix"]+ " -tp " +options["result_elastix"] 
  os.system(command_transformix)

#transform the format of the points to a 3 columns array
transformix_landmarks = transformix2np(options["transformix"]+"outputpoints.txt",300)  
transformix_landmarks = transformix_landmarks*pixel_dim


mean, std, min_error, max_error, errors= registration_errors(transformix_landmarks, points_moved)
result = np.array([mean,std])
print(mean, std, min_error, max_error)

#save in excel the results
def save_in_excel(options, img_example, result):
    excel_path = options["excel"]+'.xlsx'
    if os.path.isfile(excel_path):
        # If Workbook exists, it is updated
        rb = open_workbook(excel_path)
        r_sheet = rb.sheet_by_index(0)  # read only copy to introspect the file
        wb = copy(rb)
        w_sheet = wb.get_sheet(0)
    else:
        # Otherwise, workbook is created
        wb = Workbook()
        # add_sheet is used to create sheet.
        w_sheet = wb.add_sheet('Sheet 1')
    n = img_example[-1:]
    n = int(n)
    w_sheet.write(0, 1, "Copd1")
    w_sheet.write(0, 2, "Copd2")
    w_sheet.write(0, 3, "Copd3")
    w_sheet.write(0, 4, "Copd4")
    w_sheet.write(1, 0, "mean")
    w_sheet.write(2, 0, "std")

    for f in range(len(result)):
      w_sheet.write(f+1,n, result[f])

    wb.save(excel_path)


save_in_excel(options, img_example, result)
