import os

def fix_log():
    log_path = r"results/unet/log.csv"
    if not os.path.exists(log_path):
        print("Log not found.")
        return

    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Define the target header (Max columns)
    header = "epoch,train_loss,val_miou,val_pixel_acc,val_mean_acc,iou_road,iou_wall,iou_sign,iou_car\n"
    
    new_lines = [header]
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Skip old or duplicate headers
        if line.startswith("epoch"):
            continue
            
        parts = line.split(',')
        
        # Scenario 1: Old Schema (5 columns)
        # epoch, loss, miou, pixel_acc, mean_acc
        if len(parts) == 5:
            # Pad with 4 empty values for the class IoUs
            new_line = f"{line},,,,\n"
            new_lines.append(new_line)
            
        # Scenario 2: New Schema (9 columns)
        elif len(parts) == 9:
            new_lines.append(line + "\n")
            
        # Scenario 3: Weird Schema (8 columns - likely missing mean_acc from earlier fix)
        elif len(parts) == 8:
            # Assume it's the one missing mean_acc: 
            # epoch, loss, miou, pixel_acc, iou_road...
            # We need to insert a blank for mean_acc at index 4
            # parts: 0, 1, 2, 3, [insert], 4, 5, 6, 7
            new_parts = parts[:4] + [""] + parts[4:]
            new_lines.append(",".join(new_parts) + "\n")
            
        else:
            print(f"Skipping malformed line: {line}")

    # Write back
    with open(log_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Fixed {log_path}. Now has {len(new_lines)} lines.")

if __name__ == "__main__":
    fix_log()
