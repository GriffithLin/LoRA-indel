import torch

TextCNN = torch.load('./DeepBIO_TextCNN_model.pt')
print(TextCNN)
TextCNN.to_table('DeepBIO_TextCNN_model.txt')

# VDCNN = torch.load('./DeepBIO_VDCNN_model.pt')
# VDCNN.to_table('DeepBIO_VDCNN_model.txt')
#
# DNABERT = torch.load('./DeepBIO_6mer_DNAbert_model.pt')
# DNABERT.to_table('DeepBIO_6mer_DNAbert_model.tx t')