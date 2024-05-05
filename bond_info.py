amino_acids = {
    "GLY": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
    },
    "ALA": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
    },
    "SER": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "OG"): "single",
    },
    "THR": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "OG1"): "single",
        ("CB", "CG2"): "single",
    },
    "CYS": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "SG"): "single",
    },
    "VAL": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG1"): "single",
        ("CB", "CG2"): "single",
    },
    "LEU": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "CD1"): "single",
        ("CG", "CD2"): "single",
    },
    "ILE": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG1"): "single",
        ("CG1", "CD1"): "single",
        ("CB", "CG2"): "single",
    },
    "MET": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "SD"): "single",
        ("SD", "CE"): "single",
    },
    "PHE": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "aromatic",
        ("CG", "CD1"): "aromatic",
        ("CD1", "CE1"): "aromatic",
        ("CE1", "CZ"): "aromatic",
        ("CZ", "CE2"): "aromatic",
        ("CE2", "CD2"): "aromatic",
        ("CD2", "CG"): "aromatic",
    },
    "TYR": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "aromatic",
        ("CG", "CD1"): "aromatic",
        ("CD1", "CE1"): "aromatic",
        ("CE1", "CZ"): "aromatic",
        ("CZ", "OH"): "single",
        ("CZ", "CE2"): "aromatic",
        ("CE2", "CD2"): "aromatic",
        ("CD2", "CG"): "aromatic",
    },
    "TRP": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "aromatic",
        ("CG", "CD1"): "aromatic",
        ("CD1", "NE1"): "aromatic",
        ("NE1", "CE2"): "aromatic",
        ("CE2", "CD2"): "aromatic",
        ("CD2", "CG"): "aromatic",
        ("CE2", "CZ2"): "aromatic",
        ("CZ2", "CH2"): "aromatic",
        ("CH2", "CZ3"): "aromatic",
        ("CZ3", "CE3"): "aromatic",
        ("CE3", "CD2"): "aromatic",
    },
    "PRO": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "CD"): "single",
        ("CD", "N"): "single",
    },
    "HIS": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "aromatic",
        ("CG", "ND1"): "aromatic",
        ("ND1", "CE1"): "aromatic",
        ("CE1", "NE2"): "aromatic",
        ("NE2", "CD2"): "aromatic",
        ("CD2", "CG"): "aromatic",
    },
    "ARG": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "CD"): "single",
        ("CD", "NE"): "single",
        ("NE", "CZ"): "single",
        ("CZ", "NH1"): "double",
        ("CZ", "NH2"): "single",
    },
    "LYS": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "CD"): "single",
        ("CD", "CE"): "single",
        ("CE", "NZ"): "single",
    },
    "ASP": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "OD1"): "double",
        ("CG", "OD2"): "single",
    },
    "GLU": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "CD"): "single",
        ("CD", "OE1"): "double",
        ("CD", "OE2"): "single",
    },
    "ASN": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "OD1"): "double",
        ("CG", "ND2"): "single",
    },
    "GLN": {
        ("N", "CA"): "single",
        ("CA", "C"): "single",
        ("C", "O"): "double",
        ("CA", "CB"): "single",
        ("CB", "CG"): "single",
        ("CG", "CD"): "single",
        ("CD", "OE1"): "double",
        ("CD", "NE2"): "single",
    }
}
