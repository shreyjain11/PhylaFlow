import io
import re
import shlex
import tempfile
from typing import Optional, Dict
from Bio import AlignIO, Phylo, SeqIO
from Bio.Align.Applications import MafftCommandline
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from utils.metric_utils import calculate_norm_rf


# Built-in translate map (name -> numeric id as string). Users can supply
# a custom dictionary to `generate_nj(..., translate_map=...)` to override
# parsing from the NEXUS TRANSLATE block.
DEFAULT_TRANSLATE_MAP: Dict[str, str] = {
    "Marmota_marmota_marmota": "1",
    "Jaculus_jaculus": "2",
    "Trachypithecus_francoisi": "3",
    "Pongo_abelii": "4",
    "Cricetulus_griseus": "5",
    "Canis_lupus_dingo": "6",
    "Mesocricetus_auratus": "7",
    "Microtus_oregoni": "8",
    "Nannospalax_galili": "9",
    "Oryctolagus_cuniculus": "10",
    "Arvicola_amphibius": "11",
    "Manis_javanica": "12",
    "Ovis_aries": "13",
    "Ursus_arctos_horribilis": "14",
    "Pteropus_vampyrus": "15",
    "Pteropus_giganteus": "16",
    "Odocoileus_virginianus_texanus": "17",
    "Camelus_dromedarius": "18",
    "Condylura_cristata": "19",
    "Sus_scrofa": "20",
    "Equus_przewalskii": "21",
    "Equus_caballus": "22",
    "Equus_asinus": "23",
    "Puma_yagouaroundi": "24",
    "Balaenoptera_musculus": "25",
    "Physeter_catodon": "26",
    "Carlito_syrichta": "27",
    "Tursiops_truncatus": "28",
    "Orcinus_orca": "29",
    "Globicephala_melas": "30",
    "Phoca_vitulina": "31",
    "Tupaia_chinensis": "32",
    "Vulpes_lagopus": "33",
    "Odobenus_rosmarus_divergens": "34",
    "Ursus_maritimus": "35",
    "Puma_concolor": "36",
    "Panthera_pardus": "37",
    "Felis_catus": "38",
    "Mustela_putorius_furo": "39",
    "Ailuropoda_melanoleuca": "40",
    "Vulpes_vulpes": "41",
    "Canis_lupus_familiaris": "42",
    "Neomonachus_schauinslandi": "43",
    "Miniopterus_natalensis": "44",
    "Vicugna_pacos": "45",
    "Acinonyx_jubatus": "46",
    "Callorhinus_ursinus": "47",
    "Hyaena_hyaena": "48",
    "Eumetopias_jubatus": "49",
    "Mustela_erminea": "50",
    "Camelus_ferus": "51",
    "Ictidomys_tridecemlineatus": "52",
    "Bison_bison_bison": "53",
    "Talpa_occidentalis": "54",
    "Oryx_dammah": "55",
    "Lynx_canadensis": "56",
    "Callithrix_jacchus": "57",
    "Panthera_tigris_altaica": "58",
    "Lontra_canadensis": "59",
    "Rousettus_aegyptiacus": "60",
    "Microtus_ochrogaster": "61",
    "Bubalus_bubalis": "62",
    "Lagenorhynchus_obliquidens": "63",
    "Colobus_angolensis_palliatus": "64",
    "Mandrillus_leucophaeus": "65",
    "Theropithecus_gelada": "66",
    "Hylobates_moloch": "67",
    "Pan_troglodytes": "68",
    "Gorilla_gorilla_gorilla": "69",
    "Papio_anubis": "70",
    "Macaca_nemestrina": "71",
    "Macaca_mulatta": "72",
    "Macaca_fascicularis": "73",
    "Nomascus_leucogenys": "74",
    "Cercocebus_atys": "75",
    "Piliocolobus_tephrosceles": "76",
    "Chlorocebus_sabaeus": "77",
    "Rhinopithecus_roxellana": "78",
    "Rhinopithecus_bieti": "79",
    "Homo_sapiens": "80",
    "Pan_paniscus": "81",
    "Dipodomys_ordii": "82",
    "Cavia_porcellus": "83",
    "Leptonychotes_weddellii": "84",
    "Bos_taurus": "85",
    "Pipistrellus_kuhlii": "86",
    "Meriones_unguiculatus": "87",
    "Mus_caroli": "88",
    "Mus_musculus": "89",
    "Mus_pahari": "90",
    "Rattus_norvegicus": "91",
    "Mastomys_coucha": "92",
    "Grammomys_surdaster": "93",
    "Marmota_flaviventris": "94",
    "Hipposideros_armiger": "95",
    "Sapajus_apella": "96",
    "Capra_hircus": "97",
    "Rhinolophus_ferrumequinum": "98",
    "Ceratotherium_simum_simum": "99",
    "Bos_indicus": "100",
    "Manis_pentadactyla": "101",
    "Loxodonta_africana": "102",
    "Delphinapterus_leucas": "103",
    "Neophocaena_asiaeorientalis_asiaeorientalis": "104",
    "Phocoena_sinus": "105",
    "Chrysochloris_asiatica": "106",
    "Mirounga_leonina": "107",
    "Suricata_suricatta": "108",
    "Zalophus_californianus": "109",
    "Microcebus_murinus": "110",
    "Desmodus_rotundus": "111",
    "Artibeus_jamaicensis": "112",
    "Phyllostomus_discolor": "113",
    "Echinops_telfairi": "114",
    "Cebus_imitator": "115",
    "Lipotes_vexillifer": "116",
    "Otolemur_garnettii": "117",
    "Sturnira_hondurensis": "118",
    "Chinchilla_lanigera": "119",
    "Balaenoptera_acutorostrata_scammoni": "120",
    "Myotis_brandtii": "121",
    "Myotis_lucifugus": "122",
    "Myotis_myotis": "123",
    "Choloepus_didactylus": "124",
    "Dasypus_novemcinctus": "125",
    "Molossus_molossus": "126",
    "Aotus_nancymaae": "127",
    "Ochotona_princeps": "128",
    "Ochotona_curzoniae": "129",
    "Elephantulus_edwardii": "130",
    "Bos_mutus": "131",
    "Arvicanthis_niloticus": "132",
    "Saimiri_boliviensis_boliviensis": "133",
    "Erinaceus_europaeus": "134",
    "Myotis_davidii": "135",
    "Camelus_bactrianus": "136",
    "Sarcophilus_harrisii": "137",
    "Vombatus_ursinus": "138",
    "Phascolarctos_cinereus": "139",
    "Monodelphis_domestica": "140",
    "Trichosurus_vulpecula": "141",
    "Tachyglossus_aculeatus": "142",
    "Ornithorhynchus_anatinus": "143",
    "Sorex_araneus": "144",
    "Rattus_rattus": "145",
    "Heterocephalus_glaber": "146",
    "Fukomys_damarensis": "147",
    "Monodon_monoceros": "148",
    "Orycteropus_afer_afer": "149",
    "Onychomys_torridus": "150",
    "Propithecus_coquereli": "151",
    "Peromyscus_maniculatus_bairdii": "152",
    "Pteropus_alecto": "153",
    "Galeopterus_variegatus": "154",
    "Trichechus_manatus_latirostris": "155",
}


def _extract_taxon_number_map(nexus_path: str) -> dict:
    with open(nexus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Strip NEXUS comments [ ... ]
    text = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)
    name_to_num = {}

    in_translate = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not in_translate:
            if line.lower().startswith("translate"):
                in_translate = True
                line = line[len("translate") :].strip()
            else:
                continue
        if in_translate:
            done = False
            if ";" in line:
                line = line.replace(";", " ")
                done = True
            for chunk in line.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                parts = shlex.split(chunk)
                if len(parts) >= 2 and parts[0].isdigit():
                    name_to_num[parts[1]] = parts[0]
            if done:
                break

    if name_to_num:
        return name_to_num

    in_matrix = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not in_matrix:
            idx = line.lower().find("matrix")
            if idx == -1:
                continue
            in_matrix = True
            line = line[idx + len("matrix") :].strip()
            if not line:
                continue

        terminated = False
        if ";" in line:
            line, _sep, _after = line.partition(";")
            terminated = True

        line = line.strip()
        if not line:
            if terminated:
                break
            continue

        parts = shlex.split(line)
        if len(parts) >= 3 and parts[0].isdigit():
            name_to_num[parts[1]] = parts[0]

        if terminated:
            break

    return name_to_num


def generate_nj(
    nexus_path: str,
    distance_model: str = "identity",
    return_numbers: bool = False,
    translate_map: Optional[Dict[str, str]] = None,
) -> str:
    alignment = AlignIO.read(nexus_path, "nexus")
    unaligned = [
        SeqRecord(
            Seq(str(rec.seq).replace("-", "").replace(".", "")),
            id=rec.id,
            description="",
        )
        for rec in alignment
    ]

    with tempfile.NamedTemporaryFile("w+", suffix=".fasta") as tmp:
        SeqIO.write(unaligned, tmp, "fasta")
        tmp.flush()
        stdout, _ = MafftCommandline(input=tmp.name)()

    realigned = AlignIO.read(io.StringIO(stdout), "fasta")
    dm = DistanceCalculator(distance_model).get_distance(realigned)
    tree = DistanceTreeConstructor().nj(dm)

    if return_numbers:
        # Priority: user-supplied translate_map -> TRANSLATE block in NEXUS -> DEFAULT_TRANSLATE_MAP
        name_map = translate_map if translate_map is not None else _extract_taxon_number_map(nexus_path)

        if not name_map:
            # fall back to built-in default mapping (if present)
            name_map = DEFAULT_TRANSLATE_MAP

        if name_map:
            # ensure every sequence id from the alignment has a numeric mapping
            missing = [rec.id for rec in alignment if rec.id not in name_map]
            if missing:
                raise ValueError(
                    f"Missing numeric mapping for taxa in NEXUS file: {missing[:10]}"
                )
            for clade in tree.get_terminals():
                if clade.name in name_map:
                    clade.name = name_map[clade.name]
        else:
            if not all(str(rec.id).isdigit() for rec in alignment):
                raise ValueError(
                    "No numeric taxon mapping found in the NEXUS file. "
                    "Provide a TRANSLATE block or numeric IDs in the MATRIX, or pass `translate_map=`."
                )
    for clade in tree.get_nonterminals():
        clade.name = None

    handle = io.StringIO()
    Phylo.write(tree, handle, "newick")
    return handle.getvalue().strip()

generated_tree = generate_nj("/n/holylfs06/LABS/mzitnik_lab/Users/yektefaie/PhylaFlow/example_data/nexus/10484_NT_AL.nex", return_numbers=True, translate_map = DEFAULT_TRANSLATE_MAP)
print(generated_tree)


real_tree = "(94:5.695444e-04,(52:6.759570e-03,(((((9:3.649417e-02,((((5:8.575832e-03,7:1.021814e-02):7.871916e-03,(152:1.579362e-02,150:1.392880e-02):1.145495e-02):2.306139e-03,(11:8.140535e-03,(61:1.771256e-04,8:6.374073e-03):5.091466e-03):1.595946e-02):9.104269e-03,(87:1.990291e-02,(((132:1.830877e-02,93:1.918021e-02):1.130675e-02,(92:1.245116e-02,((88:3.138745e-03,89:6.149098e-03):4.163757e-03,90:1.980660e-02):9.446586e-03):4.814023e-03):6.191257e-03,(145:6.360993e-03,91:4.037815e-03):2.621517e-02):1.082837e-02):6.804989e-03):1.637958e-02):7.598185e-03,2:4.456109e-02):5.436358e-03,82:4.188136e-02):8.885586e-03,((32:2.767667e-02,((117:2.877956e-02,(110:7.104877e-03,151:5.613434e-03):5.806145e-03):6.482286e-03,(((10:2.037699e-02,(129:1.902382e-02,128:1.204222e-02):3.147256e-02):1.708079e-02,((((74:1.461790e-03,67:1.125601e-03):2.347179e-03,(4:5.046024e-03,(((81:1.064683e-03,68:5.420733e-04):1.672521e-03,69:1.515321e-03):4.385272e-05,80:1.637116e-03):3.755859e-03):1.264301e-03):6.411261e-04,((78:1.261508e-03,(79:1.058247e-03,3:2.146913e-03):9.421762e-04):9.483633e-04,((((71:3.719016e-04,(72:1.473210e-03,73:1.067581e-03):7.140732e-04):1.276475e-03,70:2.742526e-05):6.760221e-04,(66:1.870009e-03,(75:2.160562e-03,65:1.198734e-03):1.373228e-03):3.912370e-04):1.265023e-03,((76:2.178527e-03,64:1.710489e-03):6.162920e-04,77:3.159718e-03):7.708085e-04):8.396107e-05):5.539788e-04):5.054737e-03,(57:2.361336e-03,(127:3.465160e-03,(133:8.291864e-03,(96:3.167338e-03,115:7.727432e-04):4.053376e-03):6.653749e-04):2.966336e-05):8.056000e-03):1.075159e-02):3.040232e-03,(27:1.826288e-02,154:2.596512e-02):4.743770e-03):1.753392e-03):1.147938e-03):2.096113e-04,((((101:3.244922e-03,12:8.498502e-03):1.391261e-02,((((126:1.592938e-02,((86:1.108464e-02,((123:4.040028e-03,135:2.496977e-03):1.707405e-03,(122:2.387919e-03,121:6.562863e-04):6.109375e-04):7.362806e-03):8.738766e-03,44:1.283289e-02):2.404028e-03):3.633910e-03,((113:8.741438e-03,(112:1.177315e-02,118:1.926037e-02):1.023461e-03):1.856764e-03,111:2.012000e-02):9.371283e-03):4.715391e-03,((((15:2.821997e-03,153:3.290877e-04):2.097582e-03,16:3.697684e-04):6.779839e-03,60:3.765979e-03):9.646116e-03,(98:1.908890e-02,95:1.324285e-02):5.181010e-03):5.685107e-04):3.819621e-03,((((23:2.177383e-03,(21:5.378233e-04,22:1.040233e-03):7.642231e-04):2.205686e-02,99:1.512953e-02):8.692950e-03,(144:4.553318e-02,134:5.999143e-02):5.530795e-03):4.166052e-03,((((136:7.340459e-04,18:1.167599e-03):3.713738e-04,51:2.708505e-04):2.916930e-03,45:4.064248e-03):1.729039e-02,(20:1.510505e-02,((17:1.200972e-02,((55:5.357279e-03,(13:4.144140e-03,97:2.914317e-03):5.549442e-03):2.974042e-03,(62:2.485617e-03,(53:3.414357e-04,(131:3.284138e-03,(85:7.462456e-04,100:1.705093e-03):1.495545e-03):5.740200e-04):2.112387e-03):2.715745e-03):9.758058e-04):1.922661e-02,(((((63:1.070250e-03,29:1.270005e-03):2.500737e-04,(((148:2.556996e-05,103:1.103543e-03):1.476187e-03,(105:1.050706e-03,104:5.059574e-04):1.467135e-03):1.549365e-03,(28:5.066216e-04,30:1.015199e-03):1.662478e-03):5.997633e-05):2.278941e-03,116:3.924387e-03):2.179191e-03,26:2.563011e-03):3.019418e-03,(120:3.644086e-03,25:3.065016e-03):1.368993e-03):3.252373e-03):4.920212e-03):1.985777e-03):4.325681e-03):2.399229e-03):2.731987e-04):4.539606e-04,((19:2.697654e-02,54:1.260146e-02):2.553092e-02,((((50:1.671736e-03,39:3.175455e-04):2.428355e-03,59:2.888730e-03):5.146852e-03,((((14:1.670388e-03,35:8.319732e-05):2.525078e-03,40:6.857905e-03):1.406328e-02,(((43:4.725007e-04,(107:4.635055e-03,84:4.718747e-04):1.149699e-03):2.702797e-03,31:2.084189e-03):1.395128e-03,(34:1.019138e-03,(47:1.393448e-03,(109:4.918473e-04,49:1.505016e-04):6.548525e-04):2.429204e-03):4.286622e-03):4.547982e-03):7.838874e-04,((41:8.758035e-05,33:1.376553e-03):1.219673e-03,(42:2.209121e-05,6:1.468222e-04):1.247231e-03):1.289097e-02):1.623473e-03):2.617392e-03,((48:1.374969e-02,108:1.761641e-02):2.315226e-03,(((38:1.063670e-03,(36:3.122059e-03,(46:7.198432e-04,24:5.224009e-04):6.861359e-06):1.583038e-03):1.541784e-03,56:1.196485e-03):1.286965e-04,(58:8.796002e-04,37:1.510989e-03):1.109323e-03):8.520015e-03):9.207795e-03):8.447857e-03):1.397637e-03):3.518891e-03,((((143:2.092727e-02,142:9.360531e-03):6.985504e-02,(140:1.543795e-02,(137:2.112030e-02,((139:9.770927e-03,138:1.050299e-02):9.216951e-03,141:1.574792e-02):6.646699e-03):1.046826e-02):3.348985e-02):4.503602e-02,(124:1.337382e-02,125:1.773425e-02):8.671380e-03):1.097845e-03,((149:2.002715e-02,(106:2.749547e-02,(130:4.688535e-02,114:3.323875e-02):3.106259e-03):3.744149e-03):3.618649e-03,(155:1.069736e-02,102:1.983940e-02):5.194899e-03):8.886283e-03):4.155779e-03):3.757681e-03):2.790207e-03):4.013855e-03,((119:3.372364e-02,83:3.247474e-02):8.760347e-03,(147:2.363430e-02,146:1.373442e-02):1.403010e-02):2.091462e-02):1.931155e-02):7.079630e-03,1:8.959472e-04);"


print(calculate_norm_rf(generated_tree, real_tree))