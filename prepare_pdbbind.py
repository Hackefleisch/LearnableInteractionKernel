from tqdm import tqdm
import os.path
from enum import Enum
import traceback

from openmm.app import PDBFile
from pdbfixer import PDBFixer
from openbabel import pybel
from plip.structure.preparation import PDBComplex
from bond_info import amino_acids

import numpy as np
import torch
from torch_geometric.data import Data


pybel.ob.obErrorLog.SetOutputLevel(0)

defined_interactions = ['hydrophobic', 'hbond', 'pistacking', 'pication', 'saltbridges', 'halogenbond']

# TODO: Give credit, atom types are taken from a publication from this swedish dude
atom_types = [ 'B', 'C', 'F', 'I', 'N', 'O', 'P', 'S', 'Br', 'Cl', 'H' ]
bond_types = [ 'single', 'double', 'triple', 'aromatic' ]

# these pdbs failed processing with reasons beyond my fixing capabilities
pdb_ignore_list = [ '6rsa', '3nzc' ]

# these have ligands which include atoms which are not in the atom types list due to low representation like Fe, Cu, Si and Be
pdb_ignore_list.extend( ['1lvk', '1hyv', '1hyz', '1esz', '1k2v', '2fou', '2fov', '1y3g', '2foy', '2cfd', '2p8o', '2cfg', '2v96',
                         '3csl', '2wq4', '3m1s', '3pup', '3p3h', '3ro0', '3p44', '3p55', '3zs1', '3p3j', '3tdz', '2yak', '3qlb', 
                         '2ydm', '2z97', '3bwf', '2jld', '3cst', '3fxz', '3fy0', '3q4c', '3qo9', '4dcy', '4dcx', '3rj7', '4daw', 
                         '4ehr', '4fea', '4jhq', '4i60', '3zp9', '4jjf', '4jjg', '4kii', '3wc5', '4jfv', '4fil', '4hmq', '4jfw', 
                         '3w8o', '3whw', '4u5t', '4z46', '5fjw', '5dhf', '4xkc', '4rlp', '5ujo', '5fsb', '5m29', '5m34', '5m3b', 
                         '5hki', '5hlm', '5m2q', '5tgy', '6d1m', '5zeq', '6ceh', '6d1l', '5od5', '6hx5', '5nxx', '6b30', '5nxy',
                         '6hwz', '6pgx', '6qfv', '6abk', '6qfw', '6qfu', '6qfx', '6h3q', '6i96', '6rrm', '6frj', '6e4v', '6i97'] )

# these report a HIS atom set error
pdb_ignore_list.extend( ['1svh', '1h6e', '3s74', '3smq', '2y4k', '3bho', '3bc5', '3fxv', '3sm0', '4hai', '4ehm', '4ayt', '3rv9',
                         '3uib', '4ayw', '4gv8', '4bae', '4mb9', '4ezj', '4j8t', '4aw8', '4kv9', '4kva', '4q2k', '4ryl', '4rrr', 
                         '4rrq', '4yxu', '4zyv', '5iui', '5izu', '4xuh', '4yb6', '5hx8', '5gic', '5j3s', '5mrd', '5q0x', '5ngf',
                         '5ien', '5one', '5if6', '5wqd', '6gbx', '5wkl', '6do4', '5ypw', '6ck3', '5wgq', '6do5', '6hjk', '6ck6', 
                         '6cmr', '6cjy', '6c5t', '6qyo', '6i4x', '6edl', '6r0x', '6e0r', '6ebw'] )

class mode( Enum ):
    MOLECULE = 1
    ATOM = 2
    BOND = 3
    OTHER = 0

def load_pdbbind_receptor(pdb_code, pdbbind_path='pdbbind2020/'):
    fixer = PDBFixer( filename=pdbbind_path + pdb_code + "/" + pdb_code + "_pocket_fixed.pdb" )
    num_bonds = 0
    for res in fixer.topology.residues():
        for bond in res.internal_bonds():
            num_bonds += 1

    num_atoms = fixer.topology.getNumAtoms()

    rec_atom_labels = np.zeros( (num_atoms, len(atom_types)), dtype=int )
    rec_atom_pos = np.zeros( (num_atoms, 3), dtype=float )
    rec_bond_index = np.zeros( (2, num_bonds*2), dtype=int )
    rec_bond_labels = np.zeros( (num_bonds*2, len(bond_types)), dtype=int )

    for atom in fixer.topology.atoms():
        index = atom.index
        elem = atom.element.symbol
        # times ten because fixer stores in nanometer not angstrom
        pos_vec = fixer.positions[ index ] * 10

        rec_atom_labels[ index ][ atom_types.index( elem ) ] = 1
        for i in range( 3 ):
            rec_atom_pos[ index ][ i ] = pos_vec[ i ]._value

    skip_counter = 0

    for i, bond in enumerate(fixer.topology.bonds()):
        if bond.atom1.residue.index != bond.atom2.residue.index:
            skip_counter += 1
            continue
        i -= skip_counter
        rec_bond_index[ 0 ][ i*2 ] = bond.atom1.index
        rec_bond_index[ 1 ][ i*2 ] = bond.atom2.index
        rec_bond_index[ 1 ][ i*2+1 ] = bond.atom1.index
        rec_bond_index[ 0 ][ i*2+1 ] = bond.atom2.index
        if 'H' == bond.atom1.element.symbol or 'H' == bond.atom2.element.symbol:
            bond_type = 'single'
        else:
            if ( bond.atom1.name, bond.atom2.name ) in amino_acids[ bond.atom1.residue.name ]:
                bond_type = amino_acids[ bond.atom1.residue.name ][ ( bond.atom1.name, bond.atom2.name ) ]
            elif ( bond.atom2.name, bond.atom1.name ) in amino_acids[ bond.atom1.residue.name ]:
                bond_type = amino_acids[ bond.atom1.residue.name ][ ( bond.atom2.name, bond.atom1.name ) ]
            else:
                pass
                # TODO: Take care of these errors - it is so faronly a terminal oxygen which I want to ignore due to its infrequent appearance, but removing it would require to renumber the bonds and atoms tensors
                #print("ERROR:", pdb_code, bond.atom1.residue.name, bond.atom1.name, bond.atom2.name )
        rec_bond_labels[ i*2 ][ bond_types.index( bond_type ) ] = 1
        rec_bond_labels[ i*2 + 1 ][ bond_types.index( bond_type ) ] = 1

    torch_rec_atom_labels = torch.tensor(rec_atom_labels, dtype=torch.float)
    torch_rec_pos = torch.tensor(rec_atom_pos, dtype=torch.float)
    torch_rec_bonds = torch.tensor(rec_bond_index, dtype=torch.int64)
    torch_rec_bond_labels = torch.tensor(rec_bond_labels, dtype=torch.float)

    return Data(x=torch_rec_atom_labels, edge_index=torch_rec_bonds, edge_attr=torch_rec_bond_labels, pos=torch_rec_pos)

def load_pdbbind_ligand(pdb_code, pdbbind_path='pdbbind2020/'):
    current_mode = mode.OTHER
    natoms = 0
    nbonds = 0
    lig_atom_labels = 0
    lig_atom_pos = 0
    lig_bond_index = 0
    lig_bond_labels = 0

    with open( pdbbind_path + pdb_code + "/" + pdb_code + "_ligand.mol2" ) as file:
        for line in file:
            if line[ 0 ] == '@':
                label = line.strip().split( '>' )[ 1 ]
                if label == 'MOLECULE':
                    current_mode = mode.MOLECULE
                    next( file )
                    line = next( file ).strip().split()[:2]
                    natoms = int( line[0] )
                    nbonds = int( line[1] )
                    lig_atom_labels = np.zeros( (natoms, len(atom_types)), dtype=int )
                    lig_atom_pos = np.zeros( (natoms, 3), dtype=float )
                    lig_bond_index = np.zeros( (2, nbonds*2), dtype=int )
                    lig_bond_labels = np.zeros( (nbonds*2, len(bond_types)), dtype=int )
                elif label == 'ATOM':
                    current_mode = mode.ATOM
                    line = next( file )
                elif label == 'BOND':
                    current_mode = mode.BOND
                    line = next( file )
                else:
                    current_mode = mode.OTHER

            if current_mode == mode.ATOM:
                line = line.strip().split()
                if len( line ) >= 6:
                    elem = line[ 5 ].split('.')[0]
                    index = int( line[0] )-1
                    if elem not in atom_types:
                        raise ValueError("AtomType error:" + elem + "is not supported.")
                    lig_atom_labels[index][atom_types.index(elem)] = 1
                    for i in range( 3 ):
                        lig_atom_pos[index][i] = float( line[2+i] )

            if current_mode == mode.BOND:
                line = line.strip().split()
                if len( line ) >= 4:
                    index = int(line[0])-1
                    atom1 = int(line[1])-1
                    atom2 = int(line[2])-1

                    lig_bond_index[ 0 ][ index*2 ] = atom1
                    lig_bond_index[ 1 ][ index*2 ] = atom2
                    lig_bond_index[ 1 ][ index*2+1 ] = atom1
                    lig_bond_index[ 0 ][ index*2+1 ] = atom2

                    bond_type = ''
                    if line[3] == '1' or line[3] == 'am':
                        bond_type = 'single'
                    elif line[3] == '2':
                        bond_type = 'double'
                    elif line[3] == '3':
                        bond_type = 'triple'
                    elif line[3] == 'ar':
                        bond_type = 'aromatic'
                    elif line[3] == 'du':
                        bond_type = 'single'
                        #print("WARNING:", pdb_code, "has dummy bond in ligand file. Treating it as single bond")
                    else:
                        bond_type = 'single'
                        print("ERROR:", pdb_code, "has bond_type", line[3], "in ligand file. Treating it as single.")
                    lig_bond_labels[index*2][bond_types.index(bond_type)]=1
                    lig_bond_labels[index*2+1][bond_types.index(bond_type)]=1

    torch_lig_atom_labels = torch.tensor(lig_atom_labels, dtype=torch.float)
    torch_lig_pos = torch.tensor(lig_atom_pos, dtype=torch.float)
    torch_lig_bonds = torch.tensor(lig_bond_index, dtype=torch.int64)
    torch_lig_bond_labels = torch.tensor(lig_bond_labels, dtype=torch.float)

    return Data(x=torch_lig_atom_labels, edge_index=torch_lig_bonds, edge_attr=torch_lig_bond_labels, pos=torch_lig_pos)

def load_pdbbind_interactions(pdb_code, interaction_type, pdbbind_path='pdbbind2020/'):

    if interaction_type not in defined_interactions:
            raise ValueError("Interaction of type" + interaction_type + "is unknown.")

    protein_complex = PDBComplex()

    protein_complex.load_pdb(pdbbind_path + pdb_code + "/" + pdb_code + "_complex.pdb") # Load the PDB file into PLIP class
    protein_complex.analyze()

    if len(protein_complex.interaction_sets) == 0:
        raise Exception("No interaction partner found for " + pdb_code)
    #if len(protein_complex.interaction_sets) > 1:
    #    raise Exception("Found " + str(len(protein_complex.interaction_sets)) + " interaction partners for " + pdb_code + ". Results are ambigous for more than one partner. " + str(protein_complex.interaction_sets))
    
    all_plip_interactions = list(protein_complex.interaction_sets.values()) # Contains all interaction data

    interaction_tensors = []

    # All interactions will not include any hydrogen due to lack of consideration in PLIP + reduces number of atoms and edges
    for plip_interactions in all_plip_interactions:
        if interaction_type == "hydrophobic":
            tensor = torch.zeros((len(plip_interactions.all_hydrophobic_contacts),2), dtype=torch.int64)
            for index, interaction in enumerate(plip_interactions.all_hydrophobic_contacts):
                tensor[index,0] = interaction.bsatom_orig_idx
                tensor[index,1] = interaction.ligatom_orig_idx
            interaction_tensors.append( tensor )
        elif interaction_type == "hbond":
            tensor = torch.zeros((len(plip_interactions.all_hbonds_ldon)+len(plip_interactions.all_hbonds_pdon), 2), dtype=torch.int64)
            for index, interaction in enumerate(plip_interactions.all_hbonds_ldon):
                tensor[index,0] = interaction.a_orig_idx
                tensor[index,1] = interaction.d_orig_idx
            for index, interaction in enumerate(plip_interactions.all_hbonds_pdon):
                tensor[index + len(plip_interactions.all_hbonds_ldon),0] = interaction.d_orig_idx
                tensor[index + len(plip_interactions.all_hbonds_ldon),1] = interaction.a_orig_idx
            interaction_tensors.append( tensor )
        elif interaction_type == "pistacking":
            # TODO: Currently, this combines P and T stacking. It might improve performance to seperate them - this can be done by accessing interaction.type which either returns P or T
            tensor = torch.zeros(0, 2, dtype=torch.int64)
            for interaction in plip_interactions.pistacking:
                # I interpret pi stacking as all atoms in a ring interact with all atoms in another ring
                protein_ring_atoms = torch.Tensor(interaction.proteinring.atoms_orig_idx).int().repeat(len(interaction.ligandring.atoms_orig_idx))
                ligand_ring_atoms = torch.Tensor(interaction.ligandring.atoms_orig_idx).int().repeat_interleave(len(interaction.proteinring.atoms_orig_idx))
                tensor = torch.cat((tensor, torch.stack((protein_ring_atoms, ligand_ring_atoms), dim=0).T), dim=0)
            interaction_tensors.append( tensor )
        elif interaction_type == "pication":
            tensor = torch.zeros(0, 2, dtype=torch.int64)
            for interaction in plip_interactions.all_pi_cation_laro:
                protein_atoms = torch.Tensor(interaction.charge.atoms_orig_idx).int().repeat(len(interaction.ring.atoms_orig_idx))
                ligand_atoms = torch.Tensor(interaction.ring.atoms_orig_idx).int().repeat(len(interaction.charge.atoms_orig_idx))
                tensor = torch.cat((tensor, torch.stack((protein_atoms, ligand_atoms), dim=0).T), dim=0)

            for interaction in plip_interactions.pication_paro:
                ligand_atoms = torch.Tensor(interaction.charge.atoms_orig_idx).int().repeat(len(interaction.ring.atoms_orig_idx))
                protein_atoms = torch.Tensor(interaction.ring.atoms_orig_idx).int().repeat(len(interaction.charge.atoms_orig_idx))
                tensor = torch.cat((tensor, torch.stack((protein_atoms, ligand_atoms), dim=0).T), dim=0)
            interaction_tensors.append( tensor )
        elif interaction_type == "saltbridges":
            tensor = torch.zeros(0, 2, dtype=torch.int64)
            for interaction in plip_interactions.saltbridge_lneg:
                protein_atoms = torch.Tensor(interaction.positive.atoms_orig_idx).int().repeat(len(interaction.negative.atoms_orig_idx))
                ligand_atoms = torch.Tensor(interaction.negative.atoms_orig_idx).int().repeat(len(interaction.positive.atoms_orig_idx))
                tensor = torch.cat((tensor, torch.stack((protein_atoms, ligand_atoms), dim=0).T), dim=0)

            for interaction in plip_interactions.saltbridge_pneg:
                ligand_atoms = torch.Tensor(interaction.positive.atoms_orig_idx).int().repeat(len(interaction.negative.atoms_orig_idx))
                protein_atoms = torch.Tensor(interaction.negative.atoms_orig_idx).int().repeat(len(interaction.positive.atoms_orig_idx))
                tensor = torch.cat((tensor, torch.stack((protein_atoms, ligand_atoms), dim=0).T), dim=0)
            interaction_tensors.append( tensor )
        elif interaction_type == "halogenbond":
            tensor = torch.zeros((len(plip_interactions.halogen_bonds), 2), dtype=torch.int64)
            for index, interaction in enumerate(plip_interactions.halogen_bonds):
                tensor[index,0] = min(interaction.don_orig_idx, interaction.acc_orig_idx)
                tensor[index,1] = max(interaction.don_orig_idx, interaction.acc_orig_idx)
            interaction_tensors.append(tensor)

    interaction_tensors = torch.cat(interaction_tensors, dim=0)

    return interaction_tensors.unique(dim=0)

def prepare_pdbbind(thread_id, max_threads):
    pdb_list = []

    with open("INDEX_structure.2020") as file:
        for line in file:
            line=line.strip()
            if line[0] != "#" and line != "":
                pdb_list.append(line.split()[0])

    overwrite = True

    pdb_list = [ pdb for pdb in pdb_list if pdb not in pdb_ignore_list ]
    size = int(len(pdb_list) / max_threads)
    leftovers = len(pdb_list) % size

    pdb_list = pdb_list[thread_id*size:thread_id*size+size + (leftovers if max_threads - thread_id == 1 else 0)]
    
    errors = {}
    # collect and summarize known errors
    errors['HISset'] = []
    errors['Ligatm'] = []

    for pdb_code in tqdm(pdb_list):
        pdb_path = "pdbbind2020/" + pdb_code + "/" + pdb_code
        if os.path.isfile( pdb_path + "_combined.graph") and not overwrite:
            continue

        try:

            # --------------------------------------------------------
            # File preperations
            # --------------------------------------------------------
            fixer = PDBFixer( filename=pdb_path + "_pocket.pdb" )
            fixer.removeHeterogens(False)
            # Adding missing atoms works, but it produces highly distored atom positions - maybe due to a lack of space?
            # TODO: I will leave it out for now, but it could be an option
            #fixer.missingResidues = {}
            #fixer.findMissingAtoms()
            #fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb_path + '_pocket_fixed.pdb', 'w'))

            fixer = PDBFixer( filename=pdb_path + "_protein.pdb" )
            fixer.removeHeterogens(False)
            PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb_path + '_protein_fixed.pdb', 'w'))

            ligmol = None
            for mol in pybel.readfile('mol2', pdb_path + "_ligand.mol2"):
                if ligmol == None:
                    ligmol = mol
                else:
                    raise ValueError("ligand.mol2 contains more than one mol")
            ligmol.removeh()
            ligmol.write('pdb', pdb_path + "_ligand.pdb", overwrite=True)

            recmol = None
            for mol in pybel.readfile('pdb', pdb_path + "_protein_fixed.pdb"):
                if recmol == None:
                    recmol = mol
                else:
                    raise ValueError("fixed.pdb contains more than one mol")
            recmol.removeh()
            recmol.write('pdb', pdb_path + "_protein_fixed_noH.pdb", overwrite=True)

            pdb_str = ""
            with open( pdb_path + "_protein_fixed_noH.pdb" ) as file:
                for line in file:
                    if "ATOM" == line[:4]:
                        pdb_str += line
            with open( pdb_path + "_ligand.pdb" ) as file:
                for line in file:
                    if "ATOM" == line[:4]:
                        pdb_str += "HETATM" + line[6:]
                    elif "CONECT" == line[:6]:
                        pdb_str += line
            pdb_str += "END\n"
            with open( pdb_path + "_complex.pdb", 'w' ) as file:
                file.write( pdb_str )
            
            # --------------------------------------------------------
            # Graph calculations
            # --------------------------------------------------------

            rec_graph = load_pdbbind_receptor(pdb_code)
            lig_graph = load_pdbbind_ligand(pdb_code)
            
            combined_graph = Data(
                x = torch.cat([rec_graph.x, lig_graph.x], dim=0),
                edge_index = torch.cat([rec_graph.edge_index, lig_graph.edge_index + rec_graph.num_nodes], dim=1),
                edge_attr = torch.cat([rec_graph.edge_attr, lig_graph.edge_attr]),
                pos = torch.cat([rec_graph.pos, lig_graph.pos]),
                n_rec_nodes = [rec_graph.num_nodes],
                n_lig_nodes = [lig_graph.num_nodes],
                pdb = [pdb_code]
            )

            torch.save(combined_graph, pdb_path + "_combined.graph")
            
            # --------------------------------------------------------
            # Map preperations
            # --------------------------------------------------------
            pos_to_tensor_idx = {}
            num_rec_atoms = rec_graph.num_nodes
            for idx in range(num_rec_atoms):
                pos = tuple((rec_graph.pos[idx] * 10).int().tolist())
                pos_to_tensor_idx[pos] = idx
            for idx in range(lig_graph.num_nodes):
                pos = tuple((lig_graph.pos[idx] * 10).int().tolist())
                pos_to_tensor_idx[pos] = idx + num_rec_atoms

            pdb_idx_to_pos = {}
            pdb_rec_atoms = 0
            with open(pdb_path + "_complex.pdb") as file:
                for line in file:
                    if "ATOM" == line[:4]:
                        idx = int(line[6:11])
                        pos = (int(float(line[30:38])*10), int(float(line[38:46])*10), int(float(line[46:54])*10))
                        pdb_idx_to_pos[ idx ] = pos
                        pdb_rec_atoms += 1
                    elif "HETATM" == line[:6]:
                        idx = int(line[6:11])
                        pos = (int(float(line[30:38])*10), int(float(line[38:46])*10), int(float(line[46:54])*10))
                        pdb_idx_to_pos[ idx + pdb_rec_atoms ] = pos
            
            # --------------------------------------------------------
            # Interaction calculation and Mapping
            # --------------------------------------------------------

            for interaction_type in defined_interactions:

                interactions_t = load_pdbbind_interactions(pdb_code, interaction_type)

                # tensor split
                interactions_protein_t = interactions_t[:,0]
                interactions_ligand_t = interactions_t[:,1]

                # tensor to list
                interactions_protein = [ interactions_protein_t[i].item() for i in range(interactions_protein_t.size(0))]
                interactions_ligand = [ interactions_ligand_t[i].item() for i in range(interactions_ligand_t.size(0))]

                # pdb/interaction idx to position
                interactions_protein = [ pdb_idx_to_pos[interactions_protein[i]] for i in range(interactions_protein_t.size(0))]
                interactions_ligand = [ pdb_idx_to_pos[interactions_ligand[i]] for i in range(interactions_ligand_t.size(0))]

                # position to tensor idx
                #
                # There seems to be an issue with cutoff distances in pdbbind or plip
                # pdbbind uses 10A, plip 7.5A, but plip reports interactions for residues which are not in the _pocket.pdb file
                # I assume this is due to issues like heavy atom within cutoff or CA within cutoff
                interactions_protein_final = [ pos_to_tensor_idx[interactions_protein[i]] for i in range(interactions_protein_t.size(0))
                                         if interactions_protein[i] in pos_to_tensor_idx and interactions_ligand[i] in pos_to_tensor_idx ]
                
                interactions_ligand_final = [ pos_to_tensor_idx[interactions_ligand[i]] for i in range(interactions_ligand_t.size(0))
                                        if interactions_protein[i] in pos_to_tensor_idx and interactions_ligand[i] in pos_to_tensor_idx ]
                
                # filter interactions between differently named parts of the ligand
                interactions_protein_final_filter = [interactions_protein_final[i] for i in range(len(interactions_ligand_final))
                                                     if interactions_protein_final[i] < num_rec_atoms]
                interactions_ligand_final_filter = [interactions_ligand_final[i] for i in range(len(interactions_ligand_final))
                                                     if interactions_protein_final[i] < num_rec_atoms]

                # lists to tensors
                interactions_t = torch.stack([torch.tensor(interactions_protein_final_filter), torch.tensor(interactions_ligand_final_filter)], dim=1)

                # TODO: Write a sanity check which makes sure the mapped atom positions are identical
                #       and maybe also checks if the interactions make sense with regard to the used descriptor tensors after mapping

                torch.save(interactions_t, pdb_path + "_" + interaction_type + ".tensor")

        except Exception as e:
            if e.__str__()[:3] == "HIS":
                errors['HISset'].append(pdb_code)
            elif e.__str__()[:8] == "AtomType":
                errors['Ligatm'].append(pdb_code)
            else:
                errors[ pdb_code ] = traceback.format_exc()

    if errors['HISset'] == []:
        del errors['HISset']
    if errors['Ligatm'] == []:
        del errors['Ligatm']
    print("Found", len(errors), "Errors")
    for key in errors:
        print(key)
        print(errors[key])

def main(thread_id, max_threads):
    prepare_pdbbind(thread_id, max_threads)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare pdbbind data for training')
    parser.add_argument('--thread', type=int, required=True,
                        help='ID for the current thread for emberassing parallel execution')
    parser.add_argument('--thread_max', type=int, required=True,
                        help='Max number of used threads for emberassing parallel execution')
    
    args = parser.parse_args()
    main(args.thread, args.thread_max)