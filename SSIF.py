from bitstring import ConstBitStream
from zlib import decompress
import numpy as np
from enum import Enum

class file_packing_order(Enum):
	ZTC = 0
	ZCT = 1
	TZC = 2
	TCZ = 3
	CZT = 4
	CTZ = 5

def GetFilePackingOrder(idx):
	result = None

	for order in file_packing_order:
		if order.value == idx:
			result = order
			break

	return result

def StringFromBytes(byt):
	return str(byt.split(b'\0', 1)[0], 'utf-8').strip()

class SSIF_Reader(object):
	"""Reads .SSIF files"""
	def __init__(self, fileString):
		self.stream = ConstBitStream(filename=fileString)
		
		name_b, self.width, self.height, self.depth, self.timepoints, self.channels, self.bytesPerPix, packingOrder_idx = self.stream.readlist('bytes:64, uintle:32, uintle:32, uintle:32, uintle:32, uintle:32, uintle:32, uintle:32')
		self.packingOrder = GetFilePackingOrder(packingOrder_idx)
		self.name = StringFromBytes(name_b)

		self.channelNames = []
		for i in range(self.channels):
			self.channelNames.append(StringFromBytes(self.stream.read('bytes:32')))

		self.imageMap = []
		n = self.channels * self.depth * self.timepoints
		for i in range(n):
			nBytes = self.stream.read('uintle:32')
			self.imageMap.append({'offset': self.stream.bytepos, 'nBytes': nBytes})
			if i < (n - 1):
				self.stream.bytepos += nBytes

		def BaseLookUpFunc(first, second, third, n1, n2):
			nf = len(first)
			ns = len(second)
			nt = len(third)

			first = np.repeat(first, ns * nt)
			second = np.tile(np.repeat(second, nt), nf)
			third = np.tile(third, nf * ns)

			return (first * n1) + (second * n2) + third

		if self.packingOrder == file_packing_order['ZTC']:
			def LookUpFunc(z, t, c):
				return BaseLookUpFunc(c, t, z, self.depth * self.timepoints, self.depth)
		elif self.packingOrder == file_packing_order['ZCT']:
			def LookUpFunc(z, t, c):
				return BaseLookUpFunc(t, c, z, self.depth * self.channels, self.depth)	
		elif self.packingOrder == file_packing_order['TZC']:
			def LookUpFunc(z, t, c):
				return BaseLookUpFunc(c, z, t, self.depth * self.timepoints, self.timepoints)
		elif self.packingOrder == file_packing_order['TCZ']:
			def LookUpFunc(z, t, c):
				return BaseLookUpFunc(z, c, t, self.timepoints * self.channels, self.timepoints)
		elif self.packingOrder == file_packing_order['CZT']:
			def LookUpFunc(z, t, c):
				return BaseLookUpFunc(t, z, c, self.depth * self.channels, self.channels)
		elif self.packingOrder == file_packing_order['CTZ']:
			def LookUpFunc(z, t, c):
				return BaseLookUpFunc(z, t, c, self.timepoints * self.channels, self.channels)
		self._LookUpFunc_ = LookUpFunc

		if self.bytesPerPix == 4:
			dt = np.dtype(np.uint32)
		elif self.bytesPerPix == 2:
			dt = np.dtype(np.uint16)
		elif self.bytesPerPix == 1:
			dt = np.dtype(np.uint8)
		self._dt_ = dt.newbyteorder('<')

	def GetImage(self, z=0, t=0, c=0, image_type='raw', normalise=None):
		if type(c) is str:
			c = self.channelNames.index(c)

		if type(z) is int:
			z = [z]
		if type(t) is int:
			t = [t]
		if type(c) is int:
			c = [c]

		z = np.array(z)
		t = np.array(t)
		c = np.array(c)

		ns = self._LookUpFunc_(z, t, c)
		im = np.zeros((self.width, self.height, len(ns))).astype(self._dt_)
		for idx, n in enumerate(ns):
			offset = self.imageMap[n]['offset']
			nBytes = self.imageMap[n]['nBytes']
			self.stream.bytepos = offset

			im[:, :, idx] = np.frombuffer(decompress(self.stream.readlist('bytes:n', n=nBytes)[0], bufsize=(self.bytesPerPix * self.height * self.width)), dtype=self._dt_).reshape((self.width, self.height))

		if image_type == 'float' or image_type == 'pil':
			im = im.astype(np.float)

			if normalise == 'full_range' and image_type != 'pil':
				im /= ((2**(8*self.bytesPerPix))-1)
			elif normalise == '01' or image_type == 'pil':
				normRange = min(im.flatten()), max(im.flatten())
				im = (im - normRange[0]) / (normRange[1] - normRange[0])

			if image_type == 'pil':
				from PIL import Image
				im = [Image.fromarray(im[:, :, i] * 255.0) for i in range(im.shape[-1])]

		return im