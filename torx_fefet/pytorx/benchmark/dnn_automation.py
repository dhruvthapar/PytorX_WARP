import subprocess
import os

work_dir = os.getcwd()

def write_layer(layer_list):#0,1,2,3,4
    with open(f"{work_dir}/torx/module/w2g.py",'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if "if self.layer_count ==" in line:
            lines[i] = f"        if self.layer_count == {layer_list[0]}"
            for j in range(1,len(layer_list),1):
                lines[i] += f" or self.layer_count == {layer_list[j]}"
            lines[i] += ":\n"
            break
    with open(f"{work_dir}/torx/module/w2g.py",'w') as f:
        for line in lines:
            f.write(line)

def write_model(model):
    with open(f"{work_dir}/torx/module/fault_injection.py",'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if model == "ResNet18":
            if "#ResNet18" in line:
                lines[i] = "        #ResNet18\n"
            elif "#VGG16" in line:
                lines[i] = "        '''#VGG16\n"
        elif model == "VGG16":
            if "#VGG16" in line:
                lines[i] = "        #VGG16\n"
            elif "#ResNet18" in line:
                lines[i] = "        '''#ResNet18\n"
    with open(f"{work_dir}/torx/module/fault_injection.py",'w') as f:   
        for line in lines:
            f.write(line) 
def write_fault(fault):#pv,fc1,fc2,fc3,fc4,sap0,sapp,sapn
    with open(f"{work_dir}/torx/module/fault_injection.py",'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i]
        if "output = inject" in line:
            lines[i] = f"        output = inject_{fault}(input, input_scaled, g_dict, pv_labels, self.fault_map)\n"
            break
    with open(f"{work_dir}/torx/module/fault_injection.py",'w') as f:
        for line in lines:
            f.write(line)
write_model("ResNet18")
if 1:      
    list = ["fc1"]
    write_fault(list[0])
    write_layer([0,1,2,3,4,5])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc1.log","w"))
    write_layer([6,7,8])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc2.log","w"))
    write_layer([9,10])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc3.log","w"))
    write_layer([11])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc4.log","w"))

    write_layer([13])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc5.log","w"))
    write_layer([14])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc6.log","w"))
    write_layer([15])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc7.log","w"))
    write_layer([16])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc8.log","w"))
    write_layer([18])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc9.log","w"))
    write_layer([19])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc10.log","w"))

if 1:      
    list = ["fc2"]
    write_fault(list[0])
    write_layer([0,1,2,3,4,5])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc1.log","w"))
    write_layer([6,7,8])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc2.log","w"))
    write_layer([9,10])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc3.log","w"))
    write_layer([11])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc4.log","w"))
    write_layer([13])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc5.log","w"))
    write_layer([14])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc6.log","w"))
    write_layer([15])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc7.log","w"))
    write_layer([16])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc8.log","w"))
    write_layer([18])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc9.log","w"))
    write_layer([19])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc10.log","w"))

if 1:      
    list = ["fc3"]
    write_fault(list[0])
    write_layer([0,1,2,3,4,5])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc1.log","w"))
    write_layer([6,7,8])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc2.log","w"))
    write_layer([9,10])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc3.log","w"))
    write_layer([11])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc4.log","w"))
    write_layer([13])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc5.log","w"))
    write_layer([14])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc6.log","w"))
    write_layer([15])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc7.log","w"))
    write_layer([16])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc8.log","w"))
    write_layer([18])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc9.log","w"))
    write_layer([19])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc10.log","w"))

    #subprocess.run(["python", f"{work_dir}/main.py", "--model","vgg16"], stdout = open(f"dnn_runs/vgg_{list[0]}_{list[1]}.log","w"))

if 1:      
    list = ["fc4"]
    write_fault(list[0])
    write_layer([0,1,2,3,4,5])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc1.log","w"))
    write_layer([6,7,8])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc2.log","w"))
    write_layer([9,10])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc3.log","w"))
    write_layer([11])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc4.log","w"))
    write_layer([13])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc5.log","w"))
    write_layer([14])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc6.log","w"))
    write_layer([15])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc7.log","w"))
    write_layer([16])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc8.log","w"))
    write_layer([18])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc9.log","w"))
    write_layer([19])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc10.log","w"))

if 1:      
    list = ["sap0"]
    write_fault(list[0])
    write_layer([0,1,2,3,4,5])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc1.log","w"))
    write_layer([6,7,8])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc2.log","w"))
    write_layer([9,10])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc3.log","w"))
    write_layer([11])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc4.log","w"))
    write_layer([13])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc5.log","w"))
    write_layer([14])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc6.log","w"))
    write_layer([15])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc7.log","w"))
    write_layer([16])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc8.log","w"))
    write_layer([18])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc9.log","w"))
    write_layer([19])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc10.log","w"))

if 1:      
    list = ["sapp"]
    write_fault(list[0])
    
    write_layer([0,1,2,3,4,5])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc1.log","w"))
    write_layer([6,7,8])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc2.log","w"))
    write_layer([9,10])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc3.log","w"))
    write_layer([11])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc4.log","w"))
    write_layer([13])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc5.log","w"))
    write_layer([14])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc6.log","w"))
    write_layer([15])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc7.log","w"))
    
    write_layer([16])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc8.log","w"))
    write_layer([18])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc9.log","w"))
    
    write_layer([19])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc10.log","w"))
    
if 1:      
    list = ["sapn"]
    write_fault(list[0])
    write_layer([0,1,2,3,4,5])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc1.log","w"))
    write_layer([6,7,8])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc2.log","w"))
    write_layer([9,10])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc3.log","w"))
    write_layer([11])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc4.log","w"))
    write_layer([13])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc5.log","w"))
    write_layer([14])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc6.log","w"))
    write_layer([15])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc7.log","w"))
    write_layer([16])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc8.log","w"))
    write_layer([18])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc9.log","w"))
    write_layer([19])
    subprocess.run(["python", f"{work_dir}/main.py"], stdout = open(f"dnn_runs/resnet_{list[0]}_loc10.log","w"))

    