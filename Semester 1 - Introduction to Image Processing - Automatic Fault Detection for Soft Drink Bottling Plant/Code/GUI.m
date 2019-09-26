function varargout = GUI(varargin)
% GUI MATLAB code for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 13-Jan-2019 02:33:42

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI (see VARARGIN)
clc
rescale_factor = 1;
[I_training,I_training_avg,eig_vect_extract,proj_eigvect] =  train_all_bottles(rescale_factor);
handles.rescale_factor=rescale_factor;
handles.I_training=I_training;
handles.I_training_avg=I_training_avg;
handles.eig_vect_extract=eig_vect_extract;
handles.proj_eigvect=proj_eigvect;
guidata(hObject,handles)

imshow(0, 'Parent', handles.axes1);
imshow(0, 'Parent', handles.axes2);
imshow(0, 'Parent', handles.axes3);
imshow(0, 'Parent', handles.axes4);
imshow(0, 'Parent', handles.axes5);
imshow(0, 'Parent', handles.axes6);
imshow(0, 'Parent', handles.axes7);
% Choose default command line output for GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in segment_and_detect.
function segment_and_detect_Callback(hObject, eventdata, handles)
% hObject    handle to segment_and_detect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

    rescale_factor=handles.rescale_factor;
    I_training = handles.I_training;
    I_training_avg = handles.I_training_avg;
    eig_vect_extract = handles.eig_vect_extract;
    proj_eigvect = handles.proj_eigvect;
    [file1,path1,indx1] = uigetfile('*.jpg','Select Nifti ED Frame File');
    if isequal(file1,0)
     disp('No File was Selected')
    else
     disp(['Selecting for image: ', fullfile(path1, file1)])
     fullfile(path1, file1)
    I_in = rgb2gray(imread(fullfile(path1, file1)));
    imshow(I_in, 'Parent', handles.axes1);
    [I_testing_l,I_testing_m,I_testing_r,min_index_l,min_index_m,min_index_r] =  test_all_bottles(I_in,rescale_factor,I_training_avg,eig_vect_extract,proj_eigvect);
    imshow(I_testing_l, 'Parent', handles.axes2);
    imshow(I_testing_m, 'Parent', handles.axes3);
    imshow(I_testing_r, 'Parent', handles.axes4);
    if(I_testing_l == 0)
        imshow(0, 'Parent', handles.axes5);
    else
        imshow(squeeze(I_training(min_index_l,:,:)), 'Parent', handles.axes5);
    end
    
    if(I_testing_m == 0)
        imshow(0, 'Parent', handles.axes6);
    else
        imshow(squeeze(I_training(min_index_m,:,:)), 'Parent', handles.axes6);
    end
    
    if(I_testing_r == 0)
        imshow(0, 'Parent', handles.axes7);
    else
        imshow(squeeze(I_training(min_index_r,:,:)), 'Parent', handles.axes7);
    end
    
    if(I_testing_l == 0)
        set(handles.l_bottle, 'String', 'Error');
    elseif(min_index_l>63)
            set(handles.l_bottle, 'String', 'Normal');
    else
        score_l = ceil(min_index_l/9);
        if (score_l == 1)
            set(handles.l_bottle, 'String', 'Underfilled');
        elseif (score_l == 2)
            set(handles.l_bottle, 'String', 'Overfilled');
        elseif (score_l == 3)
            set(handles.l_bottle, 'String', 'No Label');
        elseif (score_l == 4)
            set(handles.l_bottle, 'String', 'No Label Print');
        elseif (score_l == 5)
            set(handles.l_bottle, 'String', 'Label Not Straight');
        elseif (score_l == 6)
            set(handles.l_bottle, 'String', 'Cap Missing');
        elseif (score_l == 7)
            set(handles.l_bottle, 'String', 'Deformed Bottle');
        end
    end

    if(I_testing_m == 0)
        set(handles.m_bottle, 'String', 'Error');
    elseif(min_index_m>63)
        set(handles.m_bottle, 'String', 'Normal');
    else
        score_m = ceil(min_index_m/9);
        if (score_m == 1)
            set(handles.m_bottle, 'String', 'Underfilled');
        elseif (score_m == 2)
            set(handles.m_bottle, 'String', 'Overfilled');
        elseif (score_m == 3)
            set(handles.m_bottle, 'String', 'No Label');
        elseif (score_m == 4)
            set(handles.m_bottle, 'String', 'No Label Print');
        elseif (score_m == 5)
            set(handles.m_bottle, 'String', 'Label Not Straight');
        elseif (score_m == 6)
            set(handles.m_bottle, 'String', 'Cap Missing');
        elseif (score_m == 7)
            set(handles.m_bottle, 'String', 'Deformed Bottle');
        end
    end

    if(I_testing_r == 0)
        set(handles.r_bottle, 'String', 'Error');
    elseif(min_index_r>63)
        set(handles.r_bottle, 'String', 'Normal');
    else
        score_r = ceil(min_index_r/9);
        if (score_r == 1)
            set(handles.r_bottle, 'String', 'Underfilled');
        elseif (score_r == 2)
            set(handles.r_bottle, 'String', 'Overfilled');
        elseif (score_r == 3)
            set(handles.r_bottle, 'String', 'No Label');
        elseif (score_r == 4)
            set(handles.r_bottle, 'String', 'No Label Print');
        elseif (score_r == 5)
            set(handles.r_bottle, 'String', 'Label Not Straight');
        elseif (score_r == 6)
            set(handles.r_bottle, 'String', 'Cap Missing');
        elseif (score_r == 7)
            set(handles.r_bottle, 'String', 'Deformed Bottle');
        end
    end
end
