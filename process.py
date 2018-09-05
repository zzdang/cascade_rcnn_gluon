import os

with open('./data/voc/window_file_voc07_test.txt') as f:
	content = f.read()

	images_info = content.split('#')
	images_info = [i for i in images_info if i !='' ]
	ign_info ={}
	for idx, image_info in enumerate(images_info):

		_, filepath, _, height, width, bbox_num, *bboxes_info = image_info.split('\n')
		
		filename = os.path.basename(filepath)
		bbox_num = int(bbox_num)

		bbox_info_list = []
		for bbox_idx in range(bbox_num):
			one_bbox_info = bboxes_info [ bbox_idx ]
			one_bbox_info = one_bbox_info.split(' ')
			one_bbox_info = [ int(i) for i in one_bbox_info]
			one_bbox_ign = [one_bbox_info[3],one_bbox_info[4],one_bbox_info[5],one_bbox_info[6],one_bbox_info[0],one_bbox_info[2]]
			if int(one_bbox_info[1]!=0):
				print(filename)
			bbox_info_list.append(one_bbox_ign)


		ign_info[filename]= bbox_info_list


