
import mxnet as mx 

def update_key(k):
	if k[:7]=='conv1_1':
		return  'conv0'+k[7:]
	if k[:7]=='conv1_2':
		return  'conv1'+k[7:]
	if k[:7]=='conv2_1':
		return  'conv2'+k[7:]
	if k[:7]=='conv2_2':
		return  'conv3'+k[7:]
	if k[:7]=='conv3_1':
		return  'conv4'+k[7:]
	if k[:7]=='conv3_2':
		return  'conv5'+k[7:]
	if k[:7]=='conv3_3':
		return  'conv6'+k[7:]
	if k[:7]=='conv4_1':
		return  'conv7'+k[7:]
	if k[:7]=='conv4_2':
		return  'conv8'+k[7:]
	if k[:7]=='conv4_3':
		return  'conv9'+k[7:]
	if k[:7]=='conv5_1':
		return  'conv10'+k[7:]
	if k[:7]=='conv5_2':
		return  'conv11'+k[7:]
	if k[:7]=='conv5_3':
		return  'conv12'+k[7:]
	if k[:3]=='fc6':
		return  'dense0'+k[3:]
	if k[:3]=='fc7':
		return  'dense1'+k[3:]
	else:
		return k


filename ='/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/model/VGG_16_fc2048_prune-0000.params'
filename1 ='/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/model/VGG_16_fc2048_prune_cascade.params'
loaded = [(k[4:] if k.startswith('arg:') or k.startswith('aux:') else k, v) \
          for k, v in mx.nd.load(filename).items()]
# arg_dict = { k: v for k, v in loaded}
# print(arg_dict.keys())
arg_dict = {update_key(k): v for k, v in loaded}
print(arg_dict.keys())
arg_dict['dense2_weight']= arg_dict['dense0_weight']
arg_dict['dense2_bias']= arg_dict['dense0_bias']
arg_dict['dense3_weight']= arg_dict['dense1_weight']
arg_dict['dense3_bias']= arg_dict['dense1_bias']
arg_dict['dense4_weight']= arg_dict['dense0_weight']
arg_dict['dense4_bias']= arg_dict['dense0_bias']
arg_dict['dense5_weight']= arg_dict['dense1_weight']
arg_dict['dense5_bias']= arg_dict['dense1_bias']
print(arg_dict.keys())
mx.nd.save(filename1, arg_dict)