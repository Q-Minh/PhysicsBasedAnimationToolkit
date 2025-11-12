import polyscope as ps
import polyscope.imgui as imgui
import numpy as np

def update_vdbc(vdbc, i):
    if vdbc is None:
        vdbc = np.array([i], dtype=np.int32)
        return vdbc, vdbc.shape[0]
    print("looking for index:", i)
    print("in current vdbc:")
    print(vdbc)
    found = np.where(vdbc == i)[0]
    print("we have found:")
    print(found)
    if found.shape[0] > 0:
        vdbc = np.delete(vdbc, found)
    else:
        vdbc = np.hstack([vdbc, i])
    print("new vdbc:")
    print(vdbc)
    return vdbc, vdbc.shape[0]

def select_one(hit_pos, name, cur_dirichlet):
  pick_result = ps.pick(screen_coords=hit_pos)
  #print(pick_result.is_hit, pick_result.structure_name == vm.get_name(), pick_result.structure_data['element_type'])
  if pick_result.is_hit and pick_result.structure_name == name and pick_result.structure_data['element_type'] == "vertex":
      # print(pick_result)
      i = pick_result.local_index
      cur_dirichlet, _ = update_vdbc(cur_dirichlet, i)
  return cur_dirichlet