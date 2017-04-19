require 'torchx'
require 'pl.path'
require 'lfs'
require 'image'

-- Reference: http://stackoverflow.com/questions/37657533/what-is-the-structure-of-torch-dataset

ROOT = "./data"
PRI_EXT = {"train"}
SEC_EXT = {"1", "2", "3", "4", "5"}
TARGET1 = path.join(ROOT, "train_set.t7")
--TARGET2 = path.join(ROOT, "test_set.t7")
TARGET = {TARGET1}

for k = 1, #PRI_EXT do

	data_table = {}
	label_table = {}
	src = path.join(ROOT, PRI_EXT[k])

	for i = 1, #SEC_EXT do
		print (SEC_EXT[i])
		path_local = path.join(src, SEC_EXT[i])
		print (path_local)
		files = paths.indexdir(path_local, 'jpg', true)
	
		for j = 1, files:size() do
			local img1 = image.load(files:filename(j), 1)
			table.insert(data_table, img1)
		end
	
		for j = 1, files:size() do
			table.insert(label_table, i)
		end
	end

	data_tensor = torch.Tensor(#data_table, 1, 64, 64)
	label_tensor = torch.Tensor(label_table)

	for i = 1, #data_table do
		data_tensor[i] = data_table[i]
	end

	content_of_t7 = {data = data_tensor, label = label_tensor}
	torch.save(TARGET[k], content_of_t7, 'ascii')
end

print("Required train_set.t7 files created")

--[[ train_set = torch.load(TARGET1)
test_set = torch.load(TARGET2)

setmetatable(train_set, {
	__index = function(t, i)
				return {t.data[i], t.label[i]} 
	end});

train_set.data = train_set.data:double()

function train_set:size()
	return self.data:size(1)
end

print(train_set:size())
for i = 1, train_set:size() do
	print(i)
	print(train_set.label[i])
	--image.display(train_set.data[i])	
end


setmetatable(test_set, {
	__index = function(t, i)
				return {t.data[i], t.label[i]} 
	end});

test_set.data = test_set.data:double()

function test_set:size()
	return self.data:size(1)
end

print(test_set:size())
for i = 1, test_set:size() do
	print(i)
	print(test_set.label[i])
	--image.display(test_set.data[i])	
end --]]

