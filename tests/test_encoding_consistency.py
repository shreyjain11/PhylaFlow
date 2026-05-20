import unittest
import pickle
import numpy as np
from ete3 import Tree as EteTree
from utils.bhv_distance import bhv_geodesic_with_support
from utils.random_tree import Tree
from utils.bhv_utils import (
    BHVEncoder,
    _internal_bio_clusters_from_splits,
    get_structural_polytomy_groups_from_newick,
    return_sampled_tree_boundary_decisions,
    return_sampled_tree_orthant_velocity,
    return_tree_boundary_merge_paths,
)
from utils.bhv_movie import build_tree_from_splits
from utils.metric_utils import kl_divergence_topological_distributions, compare_branch_length_distributions
from utils.utils import remove_bit
import random
from model.treeTokenizer import TreeFeatureTokenizer
import pickle


class TestEncodingConsistency(unittest.TestCase):
    """Test that encoding a tree and rebuilding it produces the same topology."""

    def assert_same_topology(self, left_newick, right_newick, message):
        left_tree = EteTree(left_newick)
        right_tree = EteTree(right_newick)
        rf_distance, max_rf, *_ = left_tree.robinson_foulds(
            right_tree,
            unrooted_trees=True,
        )
        norm_rf = 0.0 if max_rf == 0 else rf_distance / max_rf
        self.assertEqual(
            norm_rf,
            0.0,
            f"{message} RF distance={rf_distance}, max RF={max_rf}",
        )

    def setUp(self):
        self.newick_starting_trees = [
            "((103:0.00682193,((((67:0.0439808,(((((51:0.00865743,146:0.00462283):0.0222211,((148:0.0128467,37:0.0198568):0.00843991,(((143:0.0102063,142:0.00461155):0.0080365,145:0.0142993):0.0143958,147:0.00990896):0.00175042):0.00572477):0.00703723,141:0.016264):0.00474726,(((122:0.00973929,100:0.00558785):0.00661349,(57:0.016645,59:0.0172412):0.0182336):0.00153417,(12:0.00946941,(113:0.000583822,133:0.00846711):0.00483948):0.0147823):0.00574291):0.018429,144:0.362805):0.00492942):0.00471254,136:0.0418575):0.00338047,(((((28:0.0155853,29:0.0170541):0.00591746,((4:0.0212908,62:0.00963851):0.00427174,(55:0.0243675,(8:0.0272281,(35:0.0368723,17:0.041493):0.0100006):0.0040231):0.00214787):0.0123364):0.00371982,((((105:0.012147,66:0.0216846):0.0199656,((((87:0.00105789,((85:0.000775376,(96:0.00262203,72:0.00178744):0.000202943):0.00148156,(109:0.000171426,86:0.000245894):0.00260151):0.000127423):0.00238994,107:0.00194945):0.0148945,(98:0.0113757,10:0.0196428):0.00132478):0.00764489,(((88:0.0018044,101:0.000387977):0.00306796,110:0.00325975):0.00699553,(((111:0.000157476,92:2.71054e-05):0.000434327,(82:0.00261643,91:6.10849e-06):0.00118483):0.0105812,(((45:6.40702e-05,84:0.00128383):0.00315209,90:0.00350013):0.0109783,((((9:0.00133327,138:0.000320998):0.00296377,93:0.00191054):0.00111562,80:0.00219403):0.00185688,((97:0.00240807,(11:0.000364193,99:0.000115593):0.00341518):0.00268344,83:0.00228184):0.00402854):0.00317984):0.00213919):0.00107087):0.0023433):0.00858903):0.00375316,(((((((41:0.00105486,102:0.000177744):0.000947044,65:0.000806974):0.00268069,95:0.00151309):0.0147259,(68:0.0179123,((64:0.00605018,(((104:0.00232496,((2:0.000997185,139:0.00136178):0.0010471,36:0.00490643):0.000724343):0.00186945,114:0.00618397):0.00289856,(106:0.00626906,(34:0.00577039,152:0.00451465):0.00344617):0.00328261):0.00137963):0.0224077,((24:0.00116844,73:0.00453684):0.00291715,(((((115:0.000357697,((6:0.000678228,7:8.34139e-05):0.00254229,(5:0.00226796,54:4.83906e-05):0.000279059):0.00114799):0.000303177,77:0.00138509):0.000290198,(76:0.00125516,79:0.0008757):0.00149058):0.0011394,74:0.0026567):0.000569481,19:0.00534505):0.00344107):0.00296114):0.00567605):0.00680583):0.00534258,(((70:0.00055191,69:0.000526055):0.00106309,71:0.00140978):0.0205476,154:0.0143841):0.0122886):0.00110761,(((150:0.00928329,153:0.0173967):0.00650573,((63:0.00186665,(60:0.000956335,56:0.000685092):0.000501623):0.00664776,112:0.00696783):0.0129722):0.000610306,((14:0.016527,(16:0.010197,(21:0.0179435,15:0.00996396):0.00580329):0.00345656):0.011247,(30:0.0199484,((140:0.0112,(((40:0.0041253,27:0.00260998):0.00133863,26:0.000559922):0.000124735,25:0.00195663):0.00425535):0.00651433,94:0.00870227):0.00140674):0.00308169):0.00439013):0.00659369):0.000282101,(3:0.0048589,23:0.00419814):0.0133532):0.000458596):0.00275684,((50:0.0429249,((49:0.00897731,48:0.00839156):0.0727112,(46:0.015475,(42:0.0237534,(47:0.0180185,(44:0.00484563,43:0.00750911):0.00529566):0.00235894):0.0122989):0.0259945):0.0444326):0.00708893,39:0.0576919):0.00682684):0.000587665):0.00403193,(81:0.032502,((20:0.035226,(58:0.00747374,13:0.00943416):0.00539698):0.00703943,(75:0.0280917,61:0.0232738):0.00471009):0.00233203):0.00347709):0.00163097,((((31:0.00596196,108:0.00263131):0.000647815,((151:0.0014092,18:0.00274048):0.00713745,38:0.00742499):0.00115907):0.00652393,(89:0.00540574,(((((135:0.000681585,120:0.00115567):0.0027356,121:0.0007025):0.000899841,134:0.00223626):0.00183965,(((78:0.00423689,132:0.00167753):0.000377088,131:0.00106203):0.00231213,((130:0.00133489,(118:0.000230166,((123:8.25941e-05,(128:0.00206079,117:0.00398205):0.000364704):0.000441875,(124:0.00199987,(125:0.000751714,126:0.000610554):8.48961e-05):0.000559669):0.00098881):0.00166583):0.000781049,(116:0.00140055,129:0.00273853):0.000752992):0.00165537):0.00354837):0.000421227,(119:0.00121687,127:0.00182713):0.0024307):0.00143368):0.00235858):0.013371,((32:0.0113075,33:0.0185754):0.0266407,1:0.0244961):0.0143169):0.00193206):0.00374671):0.00178552,((137:0.0416873,22:0.0409797):0.00740323,(52:0.0188817,53:0.0181052):0.0103287):0.0264786):0.0293935):0.00426248,149:0.000275609,0:0.000682018);"
        ]

    def test_encoding_roundtrip_norm_rf_zero(self):
        """Test that encoding a tree and rebuilding it produces norm-RF = 0."""
        for nw in self.newick_starting_trees:
            # Parse the original tree
            t = Tree(nw)
            enc = BHVEncoder()
            masks, lens = enc.return_BHV_encoding(t)
            
            # Build the split dictionary (filter out None lengths)
            tree_dict = {m: float(l) for m, l in zip(masks, lens) if l is not None}
            n_leaves = t.n_leaves
            mapping = t.id_to_name
            
            # Rebuild the tree from splits
            _, rebuilt_newick = build_tree_from_splits(
                list(tree_dict.keys()),
                tree_dict,
                n_leaves=n_leaves,
                root_leaf=n_leaves - 1,
                mapping=mapping,
            )

            # Parse both trees with ete3 for RF comparison
            original_tree = EteTree(nw)
            rebuilt_tree = EteTree(rebuilt_newick)
            
            # Compute Robinson-Foulds distance
            rf_result = original_tree.robinson_foulds(rebuilt_tree, unrooted_trees=True)
            rf_distance = rf_result[0]
            max_rf = rf_result[1]
            
            # Compute normalized RF
            if max_rf > 0:
                norm_rf = rf_distance / max_rf
            else:
                norm_rf = 0.0
            
            # Assert that norm-RF is 0 (trees have identical topology)
            self.assertEqual(
                norm_rf, 0.0,
                f"Expected norm-RF = 0, but got {norm_rf}. "
                f"RF distance: {rf_distance}, Max RF: {max_rf}"
            )
        
    def test_velocity_consistency(self):

        real_tree_newick = '((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,45:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,51:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);'
        random_tree =  '((((((74:0.00158,67:0.00147):0.00219,(((80:0.00175,(69:0.00153,(81:0.00128,68:0.00047):0.00108):0.00021):0.00156,4:0.00497):0.00013,((133:0.00707,((115:0.00142,96:0.00162):0.00389,(127:0.00419,57:0.00234):0.00024):0.00043):0.00610,(((129:0.01483,128:0.01258):0.02349,10:0.02220):0.01067,(((110:0.01986,32:0.03018):0.00085,27:0.02439):0.00161,((84:0.02267,(90:0.05224,65:0.01999):0.06741):0.08712,((((98:0.01559,95:0.01357):0.00483,((((113:0.00996,(118:0.01509,112:0.01146):0.00201):0.00243,111:0.01759):0.00797,(126:0.01728,(((153:0.03625,135:0.03207):0.02023,((123:0.00285,(122:0.00146,121:0.00028):0.00194):0.00538,86:0.01007):0.00596):0.00314,44:0.01368):0.00247):0.00218):0.00103,(60:0.00611,(16:0.00049,15:0.00169):0.00607):0.01037):0.00035):0.00489,(((125:0.01786,124:0.01565):0.00892,(((130:0.03630,114:0.03724):0.00223,106:0.02954):0.00318,102:0.02412):0.00628):0.00240,((54:0.01483,19:0.02781):0.01776,((((45:0.00394,(136:0.00477,(51:0.00003,18:0.00041):0.00002):0.00346):0.01614,(20:0.01759,((((116:0.00391,(((105:0.00001,104:0.00042):0.00068,103:0.00237):0.00066,((63:0.00081,29:0.00050):0.00051,(30:0.00104,28:0.00070):0.00036):0.00043):0.00088):0.00037,26:0.00309):0.00137,(120:0.00521,25:0.00175):0.00206):0.00460,(17:0.00870,((62:0.00415,((131:0.00377,(100:0.00199,85:0.00976):0.00058):0.00016,53:0.00071):0.00238):0.00213,(55:0.00505,(97:0.00254,13:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.00180):0.00285,((99:0.01663,(23:0.00062,(22:0.00022,21:0.00022):0.00068):0.01808):0.00968,(101:0.00441,12:0.00560):0.01501):0.00020):0.00037,((((108:0.01911,48:0.01179):0.00251,(46:0.00155,((58:0.00043,37:0.00001):0.00143,((56:0.00191,38:0.00157):0.00046,(36:0.00140,24:0.00121):0.00020):0.00042):0.00063):0.00811):0.00699,((59:0.00365,(50:0.00067,39:0.00107):0.00288):0.00897,(((((109:0.00043,49:0.00000):0.00067,47:0.00107):0.00202,34:0.00102):0.00266,((107:0.00304,43:0.00088):0.00136,31:0.00212):0.00103):0.00509,(40:0.00414,(35:0.00000,14:0.00000):0.00326):0.00796):0.00133):0.00212):0.00123,((41:0.00009,33:0.00165):0.01313,(148:0.01747,(42:0.00000,6:0.00000):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((52:0.00623,(94:0.00106,1:0.03289):0.00421):0.02268,(((((143:0.01269,142:0.00907):0.06123,(140:0.02122,((141:0.01612,(139:0.00865,138:0.00789):0.00673):0.00139,137:0.02363):0.00457):0.02164):0.02569,(134:0.10397,(144:0.07853,(151:0.09246,(154:0.27905,117:0.06125):0.07269):0.04451):0.01766):0.00510):0.00403,((147:0.01720,146:0.01718):0.01042,(119:0.03662,83:0.02953):0.00351):0.01413):0.00066,(82:0.04391,((9:0.03393,((((145:0.01759,91:0.00547):0.02071,((132:0.01806,93:0.01632):0.00670,(92:0.01241,(89:0.00723,88:0.00408):0.01522):0.00272):0.00104):0.00989,87:0.02134):0.00196,((11:0.00845,(61:0.00226,8:0.00427):0.00417):0.01284,(150:0.03854,(7:0.00957,5:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((155:0.32836,152:0.04457):0.02833,149:0.02867):0.07311,2:0.09508):0.02500):0.00275):0.00127):0.00272):0.00096):0.00172):0.00170):0.00016):0.01078):0.00373):0.00032):0.00194,((79:0.00087,78:0.00044):0.00036,3:0.00225):0.00104):0.00028,(77:0.00227,(76:0.00200,64:0.00148):0.00034):0.00008):0.00025,(70:0.00002,66:0.00085):0.00013):0.00009,(73:0.00075,(72:0.00087,71:0.00044):0.00012):0.00022,75:0.00196):0.00000;'
        #timepoint = random.uniform(0, 1)
        timepoint = 0.6056452588709799
        newick, velocity = return_sampled_tree_orthant_velocity(
            random_tree, real_tree_newick, timepoint
        )

        tokenizer =  TreeFeatureTokenizer(
            3,
            2,
            256,
        )

        tokenization_one = tokenizer([newick])

        padded_feature, padding_mask, padded_index, leaf_mask, leaf_idx, edge_mask, edge_split_masks = tokenization_one

        # Determine num_leaves from the sampled tree
        t = Tree(newick)
        num_leaves = t.n_leaves

        # Replicate the TrainingModule velocity-vs-encoding consistency check:
        # edge_split_masks is a list (one per tree in batch); we have batch size 1
        esm = edge_split_masks[0]
        real_max_bit = max(m.bit_length() for m in esm)

        missing_splits = []
        for vel in velocity:
            adjusted_vel = vel
            if vel.bit_length() == real_max_bit + 1:
                adjusted_vel = remove_bit(vel, num_leaves - 1)
            elif vel.bit_length() > real_max_bit + 1:
                self.fail(
                    f"Velocity split mask {vel} has bit_length {vel.bit_length()} "
                    f"which exceeds real_max_bit+1 ({real_max_bit + 1})"
                )

            if adjusted_vel not in esm:
                full_mask = (1 << real_max_bit) - 1
                complement_vel = full_mask ^ adjusted_vel
                if complement_vel in esm:
                    continue
                else:
                    set_bits = [i for i in range(adjusted_vel.bit_length()) if (adjusted_vel >> i) & 1]
                    missing_splits.append((adjusted_vel, set_bits))

        self.assertEqual(
            len(missing_splits), 0,
            f"Found {len(missing_splits)} velocity split(s) not in edge_split_masks: "
            + "; ".join(
                f"split={s}, set_bits={bits}" for s, bits in missing_splits
            )
        )
    
    def test_boundary_consistency(self):
        real_tree_newick = '((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,45:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,51:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);'
        random_tree =  '((((((74:0.00158,67:0.00147):0.00219,(((80:0.00175,(69:0.00153,(81:0.00128,68:0.00047):0.00108):0.00021):0.00156,4:0.00497):0.00013,((133:0.00707,((115:0.00142,96:0.00162):0.00389,(127:0.00419,57:0.00234):0.00024):0.00043):0.00610,(((129:0.01483,128:0.01258):0.02349,10:0.02220):0.01067,(((110:0.01986,32:0.03018):0.00085,27:0.02439):0.00161,((84:0.02267,(90:0.05224,65:0.01999):0.06741):0.08712,((((98:0.01559,95:0.01357):0.00483,((((113:0.00996,(118:0.01509,112:0.01146):0.00201):0.00243,111:0.01759):0.00797,(126:0.01728,(((153:0.03625,135:0.03207):0.02023,((123:0.00285,(122:0.00146,121:0.00028):0.00194):0.00538,86:0.01007):0.00596):0.00314,44:0.01368):0.00247):0.00218):0.00103,(60:0.00611,(16:0.00049,15:0.00169):0.00607):0.01037):0.00035):0.00489,(((125:0.01786,124:0.01565):0.00892,(((130:0.03630,114:0.03724):0.00223,106:0.02954):0.00318,102:0.02412):0.00628):0.00240,((54:0.01483,19:0.02781):0.01776,((((45:0.00394,(136:0.00477,(51:0.00003,18:0.00041):0.00002):0.00346):0.01614,(20:0.01759,((((116:0.00391,(((105:0.00001,104:0.00042):0.00068,103:0.00237):0.00066,((63:0.00081,29:0.00050):0.00051,(30:0.00104,28:0.00070):0.00036):0.00043):0.00088):0.00037,26:0.00309):0.00137,(120:0.00521,25:0.00175):0.00206):0.00460,(17:0.00870,((62:0.00415,((131:0.00377,(100:0.00199,85:0.00976):0.00058):0.00016,53:0.00071):0.00238):0.00213,(55:0.00505,(97:0.00254,13:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.00180):0.00285,((99:0.01663,(23:0.00062,(22:0.00022,21:0.00022):0.00068):0.01808):0.00968,(101:0.00441,12:0.00560):0.01501):0.00020):0.00037,((((108:0.01911,48:0.01179):0.00251,(46:0.00155,((58:0.00043,37:0.00001):0.00143,((56:0.00191,38:0.00157):0.00046,(36:0.00140,24:0.00121):0.00020):0.00042):0.00063):0.00811):0.00699,((59:0.00365,(50:0.00067,39:0.00107):0.00288):0.00897,(((((109:0.00043,49:0.00000):0.00067,47:0.00107):0.00202,34:0.00102):0.00266,((107:0.00304,43:0.00088):0.00136,31:0.00212):0.00103):0.00509,(40:0.00414,(35:0.00000,14:0.00000):0.00326):0.00796):0.00133):0.00212):0.00123,((41:0.00009,33:0.00165):0.01313,(148:0.01747,(42:0.00000,6:0.00000):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((52:0.00623,(94:0.00106,1:0.03289):0.00421):0.02268,(((((143:0.01269,142:0.00907):0.06123,(140:0.02122,((141:0.01612,(139:0.00865,138:0.00789):0.00673):0.00139,137:0.02363):0.00457):0.02164):0.02569,(134:0.10397,(144:0.07853,(151:0.09246,(154:0.27905,117:0.06125):0.07269):0.04451):0.01766):0.00510):0.00403,((147:0.01720,146:0.01718):0.01042,(119:0.03662,83:0.02953):0.00351):0.01413):0.00066,(82:0.04391,((9:0.03393,((((145:0.01759,91:0.00547):0.02071,((132:0.01806,93:0.01632):0.00670,(92:0.01241,(89:0.00723,88:0.00408):0.01522):0.00272):0.00104):0.00989,87:0.02134):0.00196,((11:0.00845,(61:0.00226,8:0.00427):0.00417):0.01284,(150:0.03854,(7:0.00957,5:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((155:0.32836,152:0.04457):0.02833,149:0.02867):0.07311,2:0.09508):0.02500):0.00275):0.00127):0.00272):0.00096):0.00172):0.00170):0.00016):0.01078):0.00373):0.00032):0.00194,((79:0.00087,78:0.00044):0.00036,3:0.00225):0.00104):0.00028,(77:0.00227,(76:0.00200,64:0.00148):0.00034):0.00008):0.00025,(70:0.00002,66:0.00085):0.00013):0.00009,(73:0.00075,(72:0.00087,71:0.00044):0.00012):0.00022,75:0.00196):0.00000;'

        try:
            res = return_sampled_tree_boundary_decisions(real_tree_newick, random_tree)
        except Exception as e:
            self.fail(f"return_sampled_tree_boundary_decisions raised {type(e).__name__}: {e}")

        self.assertTrue(
            res,
            "return_sampled_tree_boundary_decisions did not produce any boundary labels to validate.",
        )

        for sample_idx, sampled_boundary in enumerate(res):
            self.assertIn(
                "newick",
                sampled_boundary,
                f"Boundary sample {sample_idx} should expose the sampled topology.",
            )
            structural_groups = {
                tuple(group)
                for group in get_structural_polytomy_groups_from_newick(
                    sampled_boundary["newick"]
                )
            }
            labels = sampled_boundary.get("labels", [])
            self.assertIsInstance(
                labels,
                list,
                f"Boundary sample {sample_idx} should expose labels as a list.",
            )
            self.assertTrue(
                labels,
                f"Boundary sample {sample_idx} should include at least one merge label.",
            )
            for merge_idx, merge_label in enumerate(labels):
                self.assertIsInstance(
                    merge_label,
                    dict,
                    f"Boundary sample {sample_idx}, merge {merge_idx} should be a label dict.",
                )
                resulting_split = merge_label.get("result_split")
                components = merge_label.get("components")
                merge_indices = merge_label.get("merge_indices")
                self.assertIsInstance(
                    resulting_split,
                    int,
                    f"Boundary sample {sample_idx}, merge {merge_idx} should return an integer split mask.",
                )
                self.assertIsInstance(
                    components,
                    list,
                    f"Boundary sample {sample_idx}, merge {merge_idx} should return components as a list.",
                )
                self.assertGreaterEqual(
                    len(components),
                    3,
                    f"Boundary sample {sample_idx}, merge {merge_idx} should expose a real polytomy with at least three components.",
                )
                self.assertIn(
                    tuple(components),
                    structural_groups,
                    f"Boundary sample {sample_idx}, merge {merge_idx} does not match any structural polytomy on the current merge-step tree.",
                )
                self.assertEqual(
                    len(components),
                    len(set(components)),
                    f"Boundary sample {sample_idx}, merge {merge_idx} contains duplicate components.",
                )
                self.assertIsInstance(
                    merge_indices,
                    list,
                    f"Boundary sample {sample_idx}, merge {merge_idx} should expose merge_indices as a list.",
                )
                self.assertGreaterEqual(
                    len(merge_indices),
                    2,
                    f"Boundary sample {sample_idx}, merge {merge_idx} must merge at least two components.",
                )
                self.assertEqual(
                    len(merge_indices),
                    len(set(merge_indices)),
                    f"Boundary sample {sample_idx}, merge {merge_idx} has duplicate merge indices.",
                )
                merged = 0
                for idx in merge_indices:
                    self.assertIsInstance(
                        idx,
                        int,
                        f"Boundary sample {sample_idx}, merge {merge_idx} should use integer merge indices.",
                    )
                    self.assertGreaterEqual(
                        idx,
                        0,
                        f"Boundary sample {sample_idx}, merge {merge_idx} has a negative merge index.",
                    )
                    self.assertLess(
                        idx,
                        len(components),
                        f"Boundary sample {sample_idx}, merge {merge_idx} has an out-of-range merge index.",
                    )
                    merged |= int(components[idx])
                self.assertEqual(
                    merged,
                    int(resulting_split),
                    f"Boundary sample {sample_idx}, merge {merge_idx} merge_indices do not reconstruct the resulting split.",
                )

    def test_boundary_merge_paths_cross_boundary(self):
        random.seed(42)

        target_tree = self.newick_starting_trees[0]
        start_tree = str(Tree(num_leaves=Tree(target_tree).n_leaves, random=True))

        start = Tree(start_tree)
        target = Tree(target_tree)
        enc = BHVEncoder()
        start_masks, start_lengths = enc.return_BHV_encoding(start)
        target_masks, target_lengths = enc.return_BHV_encoding(target)
        geodesic = bhv_geodesic_with_support(
            {mask: length for mask, length in zip(start_masks, start_lengths)},
            {mask: length for mask, length in zip(target_masks, target_lengths)},
            n_leaves=start.n_leaves,
        )

        boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
        flattened_events = return_sampled_tree_boundary_decisions(start_tree, target_tree)
        exact_flattened_events = []
        for path in boundary_paths:
            for event_idx, event in enumerate(path["events"]):
                training_labels = [
                    label for label in event["labels"] if len(label["components"]) >= 3
                ]
                if training_labels:
                    exact_flattened_events.append(
                        {
                            "newick": event["newick"],
                            "labels": training_labels,
                            "stop_after_merge": bool(
                                event_idx == (len(path["events"]) - 1)
                                and len(training_labels) == 1
                            ),
                        }
                    )

        self.assertEqual(
            len(boundary_paths),
            len(geodesic["segments"]) - 1,
            "Expected one merge path per geodesic boundary.",
        )
        self.assertEqual(
            flattened_events,
            exact_flattened_events,
            "Training-facing boundary samples should exactly match the exact structural boundary events.",
        )

        saw_boundary_labels = False
        for path in boundary_paths:
            boundary_idx = path["boundary_index"]
            boundary_lengths = {
                int(mask): float(length)
                for mask, length in geodesic["segments"][boundary_idx]["end_lengths"].items()
                if length > 1e-8
            }
            current_clusters = _internal_bio_clusters_from_splits(
                boundary_lengths.keys(),
                start.n_leaves,
            )
            final_clusters = _internal_bio_clusters_from_splits(
                list(boundary_lengths.keys()) + [
                    int(split) for split in geodesic["segments"][boundary_idx]["Bi"]
                ],
                start.n_leaves,
            )
            expected_births = set(final_clusters - current_clusters)

            self.assertEqual(
                set(path["births"]),
                expected_births,
                f"Boundary {boundary_idx} returned births do not match the post-boundary refinement.",
            )

            _, expected_start_newick = build_tree_from_splits(
                list(boundary_lengths.keys()),
                boundary_lengths,
                n_leaves=start.n_leaves,
                root_leaf=start.n_leaves - 1,
                mapping=start.id_to_name,
            )
            self.assert_same_topology(
                path["start_newick"],
                expected_start_newick,
                f"Boundary {boundary_idx} start topology does not match the pre-boundary tree.",
            )

            expected_end_lengths = dict(boundary_lengths)
            for split in expected_births:
                expected_end_lengths[int(split)] = 0.1
            _, expected_end_newick = build_tree_from_splits(
                list(expected_end_lengths.keys()),
                expected_end_lengths,
                n_leaves=start.n_leaves,
                root_leaf=start.n_leaves - 1,
                mapping=start.id_to_name,
            )
            self.assert_same_topology(
                path["end_newick"],
                expected_end_newick,
                f"Boundary {boundary_idx} end topology does not match the post-boundary tree.",
            )

            current_newick = path["start_newick"]
            returned_splits = []
            for event_idx, event in enumerate(path["events"]):
                self.assert_same_topology(
                    current_newick,
                    event["newick"],
                    f"Boundary {boundary_idx}, event {event_idx} does not start from the current boundary topology.",
                )

                current_tree = Tree(current_newick)
                current_masks, current_lengths = enc.return_BHV_encoding(current_tree)
                current_mask_set = {int(mask) for mask in current_masks}
                current_full = 0
                for mask in current_masks:
                    current_full |= int(mask)
                updated_lengths = {
                    int(mask): float(length)
                    for mask, length in zip(current_masks, current_lengths)
                    if length is not None and length > 1e-8
                }

                for merge_idx, merge_label in enumerate(event["labels"]):
                    saw_boundary_labels = True
                    resulting_split = int(merge_label["result_split"])
                    components = [int(component) for component in merge_label["components"]]
                    merge_indices = [int(idx) for idx in merge_label["merge_indices"]]
                    self.assertGreaterEqual(
                        len(components),
                        2,
                        f"Boundary {boundary_idx}, event {event_idx}, merge {merge_idx} must expose at least two current components.",
                    )
                    self.assertEqual(
                        len(components),
                        len(set(components)),
                        f"Boundary {boundary_idx}, event {event_idx}, merge {merge_idx} contains duplicate components.",
                    )
                    self.assertGreaterEqual(
                        len(merge_indices),
                        2,
                        f"Boundary {boundary_idx}, event {event_idx}, merge {merge_idx} must merge at least two components.",
                    )
                    self.assertEqual(
                        len(merge_indices),
                        len(set(merge_indices)),
                        f"Boundary {boundary_idx}, event {event_idx}, merge {merge_idx} has duplicate merge indices.",
                    )

                    merged = 0
                    for component in components:
                        self.assertTrue(
                            component in current_mask_set
                            or (current_full and (current_full ^ component) in current_mask_set),
                            f"Boundary {boundary_idx}, event {event_idx}, merge {merge_idx} uses component {component} that is not present in the current tree.",
                        )
                    for idx in merge_indices:
                        self.assertGreaterEqual(
                            idx,
                            0,
                            f"Boundary {boundary_idx}, event {event_idx}, merge {merge_idx} has a negative merge index.",
                        )
                        self.assertLess(
                            idx,
                            len(components),
                            f"Boundary {boundary_idx}, event {event_idx}, merge {merge_idx} has an out-of-range merge index.",
                        )
                        merged |= components[idx]

                    self.assertEqual(
                        merged,
                        int(resulting_split),
                        f"Boundary {boundary_idx}, event {event_idx}, merge {merge_idx} merge_indices do not reconstruct the resulting split.",
                    )

                    returned_splits.append(int(resulting_split))
                    updated_lengths[int(resulting_split)] = 0.1

                _, current_newick = build_tree_from_splits(
                    list(updated_lengths.keys()),
                    updated_lengths,
                    n_leaves=current_tree.n_leaves,
                    root_leaf=current_tree.n_leaves - 1,
                    mapping=current_tree.id_to_name,
                )

            self.assertEqual(
                set(returned_splits),
                expected_births,
                f"Boundary {boundary_idx} returned splits do not match the post-boundary births.",
            )
            self.assert_same_topology(
                current_newick,
                path["end_newick"],
                f"Boundary {boundary_idx} merge path did not reach the tree on the other side of the boundary.",
            )

            if not expected_births:
                self.assertEqual(
                    path["events"],
                    [],
                    f"Boundary {boundary_idx} should not emit merge events when no new clusters are born.",
                )

        self.assertTrue(
            saw_boundary_labels,
            "return_tree_boundary_merge_paths did not produce any boundary labels to validate.",
        )
    
    def test_td2_tokenization_consistency(self):
        missing_split = 45671925826308349272926687559009236932479680511
        newick_start_tree = '((((((127:0.00158,119:0.00147):0.00219,(((134:0.00175,(121:0.00153,(135:0.00128,120:0.00047):0.00108):0.00021):0.00156,89:0.00497):0.00013,((38:0.00707,((18:0.00142,151:0.00162):0.00389,(31:0.00419,108:0.00234):0.00024):0.00043):0.0061,(((33:0.01483,32:0.01258):0.02349,1:0.0222):0.01067,(((13:0.01986,81:0.03018):0.00085,75:0.02439):0.00161,((138:0.02267,(145:0.05224,117:0.01999):0.06741):0.08712,((((153:0.01559,150:0.01357):0.00483,((((16:0.00996,(21:0.01509,15:0.01146):0.00201):0.00243,14:0.01759):0.00797,(30:0.01728,(((60:0.03625,40:0.03207):0.02023,((27:0.00285,(26:0.00146,25:0.00028):0.00194):0.00538,140:0.01007):0.00596):0.00314,94:0.01368):0.00247):0.00218):0.00103,(112:0.00611,(63:0.00049,56:0.00169):0.00607):0.01037):0.00035):0.00489,(((29:0.01786,28:0.01565):0.00892,(((35:0.0363,17:0.03724):0.00223,8:0.02954):0.00318,4:0.02412):0.00628):0.0024,((105:0.01483,66:0.02781):0.01776,((((95:0.00394,(41:0.00477,(102:3e-05,65:0.00041):2e-05):0.00346):0.01614,(68:0.01759,((((19:0.00391,(((7:1e-05,6:0.00042):0.00068,5:0.00237):0.00066,((115:0.00081,77:0.0005):0.00051,(79:0.00104,76:0.0007):0.00036):0.00043):0.00088):0.00037,74:0.00309):0.00137,(24:0.00521,73:0.00175):0.00206):0.0046,(64:0.0087,((114:0.00415,((36:0.00377,(2:0.00199,139:0.00976):0.00058):0.00016,104:0.00071):0.00238):0.00213,(106:0.00505,(152:0.00254,34:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.0018):0.00285,((154:0.01663,(71:0.00062,(70:0.00022,69:0.00022):0.00068):0.01808):0.00968,(3:0.00441,23:0.0056):0.01501):0.0002):0.00037,((((10:0.01911,98:0.01179):0.00251,(96:0.00155,((109:0.00043,86:1e-05):0.00143,((107:0.00191,87:0.00157):0.00046,(85:0.0014,72:0.00121):0.0002):0.00042):0.00063):0.00811):0.00699,((110:0.00365,(101:0.00067,88:0.00107):0.00288):0.00897,(((((11:0.00043,99:0):0.00067,97:0.00107):0.00202,83:0.00102):0.00266,((9:0.00304,93:0.00088):0.00136,80:0.00212):0.00103):0.00509,(90:0.00414,(84:0,45:0):0.00326):0.00796):0.00133):0.00212):0.00123,((91:9e-05,82:0.00165):0.01313,(54:0.01747,(92:0,111:0):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((103:0.00623,(149:0.00106,0:0.03289):0.00421):0.02268,(((((49:0.01269,48:0.00907):0.06123,(46:0.02122,((47:0.01612,(44:0.00865,43:0.00789):0.00673):0.00139,42:0.02363):0.00457):0.02164):0.02569,(39:0.10397,(50:0.07853,(58:0.09246,(61:0.27905,20:0.06125):0.07269):0.04451):0.01766):0.0051):0.00403,((53:0.0172,52:0.01718):0.01042,(22:0.03662,137:0.02953):0.00351):0.01413):0.00066,(136:0.04391,((144:0.03393,((((51:0.01759,146:0.00547):0.02071,((37:0.01806,148:0.01632):0.0067,(147:0.01241,(143:0.00723,142:0.00408):0.01522):0.00272):0.00104):0.00989,141:0.02134):0.00196,((12:0.00845,(113:0.00226,133:0.00427):0.00417):0.01284,(57:0.03854,(122:0.00957,100:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((62:0.32836,59:0.04457):0.02833,55:0.02867):0.07311,67:0.09508):0.025):0.00275):0.00127):0.00272):0.00096):0.00172):0.0017):0.00016):0.01078):0.00373):0.00032):0.00194,((132:0.00087,131:0.00044):0.00036,78:0.00225):0.00104):0.00028,(130:0.00227,(129:0.002,116:0.00148):0.00034):8e-05):0.00025,(123:2e-05,118:0.00085):0.00013):9e-05,(126:0.00075,(125:0.00087,124:0.00044):0.00012):0.00022,128:0.00196);'
        problematic_td2 = pickle.load(open("tests/problematic_td2.pickle", "rb"))
        n_leaves = 156
        mapp = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19', 20: '20', 21: '21', 22: '22', 23: '23', 24: '24', 25: '25', 26: '26', 27: '27', 28: '28', 29: '29', 30: '30', 31: '31', 32: '32', 33: '33', 34: '34', 35: '35', 36: '36', 37: '37', 38: '38', 39: '39', 40: '40', 41: '41', 42: '42', 43: '43', 44: '44', 45: '45', 46: '46', 47: '47', 48: '48', 49: '49', 50: '50', 51: '51', 52: '52', 53: '53', 54: '54', 55: '55', 56: '56', 57: '57', 58: '58', 59: '59', 60: '60', 61: '61', 62: '62', 63: '63', 64: '64', 65: '65', 66: '66', 67: '67', 68: '68', 69: '69', 70: '70', 71: '71', 72: '72', 73: '73', 74: '74', 75: '75', 76: '76', 77: '77', 78: '78', 79: '79', 80: '80', 81: '81', 82: '82', 83: '83', 84: '84', 85: '85', 86: '86', 87: '87', 88: '88', 89: '89', 90: '90', 91: '91', 92: '92', 93: '93', 94: '94', 95: '95', 96: '96', 97: '97', 98: '98', 99: '99', 100: '100', 101: '101', 102: '102', 103: '103', 104: '104', 105: '105', 106: '106', 107: '107', 108: '108', 109: '109', 110: '110', 111: '111', 112: '112', 113: '113', 114: '114', 115: '115', 116: '116', 117: '117', 118: '118', 119: '119', 120: '120', 121: '121', 122: '122', 123: '123', 124: '124', 125: '125', 126: '126', 127: '127', 128: '128', 129: '129', 130: '130', 131: '131', 132: '132', 133: '133', 134: '134', 135: '135', 136: '136', 137: '137', 138: '138', 139: '139', 140: '140', 141: '141', 142: '142', 143: '143', 144: '144', 145: '145', 146: '146', 147: '147', 148: '148', 149: '149', 150: '150', 151: '151', 152: '152', 153: '153', 154: '154', 155: 'ROOT_DUMMY'}
        graph, td2_newick = build_tree_from_splits(
                            list(problematic_td2.keys()),
                            problematic_td2,
                            n_leaves,
                            root_leaf=n_leaves - 1,
                            mapping=mapp,
                        )

        tokenizer =  TreeFeatureTokenizer(
            3,
            2,
            256,
        )

        tokenized_trees = tokenizer([td2_newick])
        edge_mask = tokenized_trees[-1][0]

        #Assert missing split is in edge_mask 
        self.assertIn(missing_split, problematic_td2, f"Missing split {missing_split} not found in problematic_td2")
        self.assertIn(missing_split, edge_mask, f"Missing split {missing_split} not found in edge_mask")



    # def test_sampled_trees_metrics(self):
    #     """Test average norm-RF and KL divergence between sampled and posterior trees."""
    #     # Load the sampled trees
    #     sampled, posterior = pickle.load(open("samples/sample_trees_100.pkl", "rb"))
        
    #     # Compute average normalized RF between sampled and posterior trees
    #     rf_distances = []
    #     n_pairs = min(len(sampled), len(posterior))
        
    #     for i in range(n_pairs):
    #         try:
    #             t1 = EteTree(sampled[i])
    #             t2 = EteTree(posterior[i])
    #             rf_result = t1.robinson_foulds(t2, unrooted_trees=True)
    #             rf_distance = rf_result[0]
    #             max_rf = rf_result[1]
    #             if max_rf > 0:
    #                 norm_rf = rf_distance / max_rf
    #             else:
    #                 norm_rf = 0.0
    #             rf_distances.append(norm_rf)
    #         except Exception as e:
    #             print(f"Error computing RF for pair {i}: {e}")
    #             continue
        
    #     avg_norm_rf = np.mean(rf_distances)
    #     std_norm_rf = np.std(rf_distances)
        
    #     print(f"\n=== Sampled Trees (100) Metrics ===")
    #     print(f"Average norm-RF: {avg_norm_rf:.4f} ± {std_norm_rf:.4f}")
    #     print(f"Number of pairs compared: {len(rf_distances)}")
        
    #     # Compute KL divergence between topological distributions
    #     # Determine num_leaves from the first tree
    #     t = Tree(sampled[0])
    #     num_leaves = t.n_leaves
        
    #     kl_result = kl_divergence_topological_distributions(
    #         sampled, posterior, num_leaves=num_leaves
    #     )
    #     kl_div = kl_result['kl_divergence_topological']
        
    #     print(f"KL divergence (topological): {kl_div:.6f}")
        
    #     # Basic sanity checks
    #     self.assertGreaterEqual(avg_norm_rf, 0.0, "Average norm-RF should be >= 0")
    #     self.assertLessEqual(avg_norm_rf, 1.0, "Average norm-RF should be <= 1")
    #     self.assertLessEqual(kl_div, 10, "KL divergence should be >= 0")

    #     # res = compare_branch_length_distributions(posterior, sampled)
    #     # self.assertLessEqual(res['kl_divergence_branch_length'], 10)
    #     # self.assertLessEqual(res['js_divergence_branch_length'], 10)

        


if __name__ == "__main__":
    unittest.main()
