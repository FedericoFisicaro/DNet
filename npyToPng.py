import numpy as np
from PIL import Image
import os
import cv2


models = ["Umons_AllOF-AllH-50","Umons_AllOF-H1-50","Umons_AllOF-H2-50","Umons_AllOF-H3-50","Umons_OF1-AllH-50","Umons_OF1-H1-50","Umons_OF1-H2-50","Umons_OF1-H3-50"]
splits = ["obj","ref"]

for model in models :
    print("--------------------------------------- \n---------------------------------------")
    print(model)
    for split in splits:
        print("-------------")
        print(split)
        print("-------------")

        inputNPY = "results/" + model + "/disps_" + split +"_split.npy"

        savedir = "results/"+ model + "/" + split

        test_files_list_path = "/home/FisicaroF/3/DNet/splits/" + split + "/test_files.txt"

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        predictions = np.load(inputNPY)

        test_files_list = open(test_files_list_path, "r")
        test_files = test_files_list.read().splitlines()
        test_files_list.close()

        for i,prediction in enumerate(predictions):
            if i % 100 == 0:
                print(i)
            # image = Image.fromarray(prediction)
            # image = image.convert("L")
            # image.save(os.path.join(savedir,str(i)+".png"))
            file_path = test_files[i].split()[0].split("/")
            file_name = file_path[0] + "_" + file_path[1] + "_" + file_path[2] + "_" + file_path[-1].split(".")[0] 


            disp_resized = cv2.resize(prediction, (1280, 544))
            # depth = STEREO_SCALE_FACTOR / disp_resized
            depth = 1 / disp_resized
            depth = np.clip(depth, 0, 20)
            depth = depth * 1000.0
            depth = depth.astype(np.uint16)
            save_path = os.path.join(savedir, file_name +".png")
            
            cv2.imwrite(save_path, depth, [cv2.IMWRITE_PNG_COMPRESSION, 0]) 