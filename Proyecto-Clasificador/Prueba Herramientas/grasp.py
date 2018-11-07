import os

def displayDataset(datasetDir=None):
    #So If you want to read dataset left_positive, datasetDir is PATH_TO_DATASET/left_positive.
    #If the folders Code and DATASET is in the same folder,
    #displayDataset(../DATASET/left_positive) should run

    #This code should display all the interactions. Requires a recent version
    #of matlab for drawRectangleonImageAtAngle.m to run.
    
    
    folderContents = dir(datasetDir)
    folderContents = folderContents(mslice[3:end])

    for interactionLooper in mslice[1:length(folderNames)]:
        interactionPath = fullfile(datasetDir, folderNames(interactionLooper))
        ImageFolderPath = fullfile(interactionPath, mstring('Images'))
        interactionDataFile = fullfile(interactionPath, mstring('dataInfo.txt')); print interactionDataFile
        labelF = fopen(interactionDataFile, mstring('r'))
        dataLine = fgetl(labelF)    # Since the first line only contains the text layout of the file
        dataLine = fgetl(labelF)

        while ischar(dataLine):
            dataFileCell = strsplit(dataLine, mstring(','))
            ImageName = dataFileCell(1)
            center_h = int16(str2num(dataFileCell(2)))
            center_w = int16(str2num(dataFileCell(3)))
            theta = double(str2num(dataFileCell(4)))
            p_width = double(str2num(dataFileCell(5)))
            p_height = double(str2num(dataFileCell(6)))

            ImagePath = fullfile(ImageFolderPath, ImageName)
            I = imread(ImagePath)
            outimg = drawRectangleonImageAtAngle(I, mcat([center_h, OMPCSEMI, center_w]), p_width, p_height, -theta * 180 / pi)

            figure(1)
            hold(mstring('off'))
            imshow(outimg)
            dataLine = fgetl(labelF)



def drawRectangleonImageAtAngle(img=None, center=None, width=None, height=None, angle=None):

    # hdl = imshow(img); hold on;
    theta = angle * (pi / 180)
    coords = mcat([center(1) - (width / 2), center(1) - (width / 2), center(1) + (width / 2), center(1) + (width / 2), OMPCSEMI, center(2) - (height / 2), center(2) + (height / 2), center(2) + (height / 2), center(2) - (height / 2)])
    R = mcat([cos(theta), sin(theta), OMPCSEMI, -sin(theta), cos(theta)])
    rot_coords = R * double(coords - repmat(center, mcat([1, 4]))) + double(repmat(center, mcat([1, 4])))
    rot_coords(mslice[:], 5).lvalue = rot_coords(mslice[:], 1)
    l1 = mcat([rot_coords(mslice[:], 1).cT, rot_coords(mslice[:], 2).cT])
    l2 = mcat([rot_coords(mslice[:], 2).cT, rot_coords(mslice[:], 3).cT])
    l3 = mcat([rot_coords(mslice[:], 3).cT, rot_coords(mslice[:], 4).cT])
    l4 = mcat([rot_coords(mslice[:], 4).cT, rot_coords(mslice[:], 1).cT])

    shapeInserterGreenLine = vision.ShapeInserter(mstring('Shape'), mstring('Lines'), mstring('BorderColor'), mstring('Custom'), mstring('LineWidth'), 10, mstring('CustomBorderColor'), mcat([0, 255, 0]))
    shapeInserterRedLine = vision.ShapeInserter(mstring('Shape'), mstring('Lines'), mstring('BorderColor'), mstring('Custom'), mstring('LineWidth'), 10, mstring('CustomBorderColor'), mcat([255, 0, 0]))
    outimg = img
    outimg = step(shapeInserterGreenLine, outimg, int16(mcat([l1, OMPCSEMI, l3])))
    outimg = step(shapeInserterRedLine, outimg, int16(mcat([l2, OMPCSEMI, l4])))

displayDataset('left_positive')