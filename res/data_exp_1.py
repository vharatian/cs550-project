epochs_exp_1 = ['50', '200', '500']
lr_exp_1 = ['0.1', '0.01', '0.001']
num_hidden_units_exp_1 = ['1', '2', '4', '8']
num_hidden_layers_exp_1 = ['1', '2', '3', '4']
initial_weights_exp_1= ['glorot_normal', 'random_normal']

data_exp_1_arr = [[0.4641379498141517, 3, 1, 0.001, 'random_normal', 50],
 [0.4641379498141517, 3, 1, 0.001, 'random_normal', 200],
 [0.4641379498141517, 3, 1, 0.001, 'random_normal', 500],
 [0.4641379498141517, 4, 1, 0.001, 'random_normal', 50],
 [0.4641379498141517, 4, 1, 0.001, 'random_normal', 200],
 [0.4641379498141517, 4, 1, 0.001, 'random_normal', 500],
 [0.4727215543180797, 4, 8, 0.1, 'glorot_normal', 50],
 [0.4780222473348129, 4, 4, 0.1, 'random_normal', 50],
 [0.47835282584720956, 4, 1, 0.01, 'random_normal', 50],
 [0.47835282584720956, 4, 1, 0.01, 'random_normal', 500],
 [0.47835282584720956, 4, 8, 0.1, 'random_normal', 500],
 [0.4794984138222119, 3, 2, 0.01, 'random_normal', 50],
 [0.47967513989679633, 4, 4, 0.1, 'glorot_normal', 500],
 [0.48280419894617876, 3, 4, 0.1, 'glorot_normal', 50],
 [0.48280419894617876, 4, 4, 0.1, 'random_normal', 200],
 [0.4841379498141517, 3, 1, 0.01, 'random_normal', 50],
 [0.48512968535134177, 3, 1, 0.01, 'random_normal', 200],
 [0.4858842430596212, 3, 1, 0.01, 'glorot_normal', 200],
 [0.48711315642572195, 4, 1, 0.1, 'random_normal', 50],
 [0.4872784456819203, 3, 1, 0.001, 'glorot_normal', 200],
 [0.4872784456819203, 3, 2, 0.001, 'random_normal', 200],
 [0.4877743134505154, 3, 1, 0.01, 'random_normal', 500],
 [0.48792816588832755, 3, 4, 0.1, 'glorot_normal', 500],
 [0.4887660489877055, 4, 8, 0.1, 'glorot_normal', 500],
 [0.4904189415496889, 3, 1, 0.1, 'glorot_normal', 50],
 [0.49058423080588726, 4, 1, 0.001, 'glorot_normal', 50],
 [0.49058423080588726, 4, 1, 0.01, 'random_normal', 200],
 [0.4912339510122946, 4, 1, 0.1, 'glorot_normal', 500],
 [0.4912339510122946, 4, 2, 0.01, 'random_normal', 200],
 [0.49141067708687897, 3, 8, 0.1, 'random_normal', 50],
 [0.49156452952469126, 4, 2, 0.01, 'random_normal', 500],
 [0.4917412555992756, 3, 2, 0.1, 'random_normal', 50],
 [0.49207183411167243, 3, 1, 0.001, 'glorot_normal', 50],
 [0.49256770188026744, 4, 1, 0.1, 'glorot_normal', 200],
 [0.4932288589050608, 3, 1, 0.001, 'glorot_normal', 500],
 [0.49339414816125915, 4, 2, 0.001, 'random_normal', 50],
 [0.49339414816125915, 4, 2, 0.001, 'random_normal', 500],
 [0.49387857911146804, 4, 1, 0.1, 'glorot_normal', 50],
 [0.49389001592985415, 3, 4, 0.1, 'glorot_normal', 200],
 [0.49389001592985426, 2, 1, 0.1, 'glorot_normal', 500],
 [0.49389001592985426, 3, 1, 0.1, 'glorot_normal', 200],
 [0.4940438683676664, 4, 4, 0.1, 'random_normal', 500],
 [0.4940553051860525, 4, 4, 0.001, 'random_normal', 500],
 [0.49422059444225086, 3, 2, 0.1, 'glorot_normal', 500],
 [0.4943744468800631, 3, 2, 0.1, 'glorot_normal', 50],
 [0.4943858836984492, 3, 2, 0.001, 'random_normal', 500],
 [0.49455117295464757, 4, 2, 0.01, 'random_normal', 50],
 [0.49455117295464757, 4, 8, 0.1, 'glorot_normal', 200],
 [0.4952008931610548, 4, 2, 0.1, 'random_normal', 200],
 [0.4952123299794409, 4, 4, 0.01, 'random_normal', 500],
 [0.4955429084918377, 4, 1, 0.1, 'random_normal', 200],
 [0.49569676092965, 3, 8, 0.1, 'glorot_normal', 200],
 [0.49653464402902775, 3, 2, 0.1, 'glorot_normal', 200],
 [0.4968652225414244, 4, 2, 0.1, 'glorot_normal', 200],
 [0.4980108105164268, 2, 1, 0.1, 'glorot_normal', 50],
 [0.4986834043596062, 4, 2, 0.001, 'random_normal', 200],
 [0.4991678353098152, 3, 4, 0.1, 'random_normal', 500],
 [0.49934456138439964, 4, 1, 0.01, 'glorot_normal', 200],
 [0.49950985064059805, 4, 4, 0.001, 'random_normal', 200],
 [0.49984042915299476, 4, 2, 0.1, 'random_normal', 50],
 [0.5013165956403937, 3, 8, 0.1, 'random_normal', 200],
 [0.5016471741527904, 2, 4, 0.1, 'random_normal', 500],
 [0.5023197679959699, 3, 1, 0.1, 'random_normal', 500],
 [0.5029809250207633, 3, 8, 0.1, 'glorot_normal', 50],
 [0.5034653559709723, 4, 2, 0.1, 'random_normal', 500],
 [0.5047991068389451, 4, 8, 0.1, 'random_normal', 200],
 [0.5051296853513418, 2, 1, 0.1, 'random_normal', 200],
 [0.5064405625825424, 2, 1, 0.1, 'random_normal', 500],
 [0.5097577845248955, 4, 4, 0.1, 'glorot_normal', 200],
 [0.5107495200620857, 2, 1, 0.1, 'glorot_normal', 200],
 [0.5120718341116725, 4, 8, 0.1, 'random_normal', 50],
 [0.5122256865494845, 3, 4, 0.1, 'random_normal', 50],
 [0.5128868435742782, 3, 4, 0.1, 'random_normal', 200],
 [0.5135594374174575, 4, 1, 0.001, 'glorot_normal', 200],
 [0.5142091576238649, 3, 1, 0.1, 'random_normal', 200],
 [0.5142205944422509, 3, 1, 0.1, 'glorot_normal', 500],
 [0.5145511729546476, 4, 4, 0.01, 'random_normal', 200],
 [0.5147164622108459, 4, 1, 0.01, 'glorot_normal', 50],
 [0.5152008931610549, 3, 8, 0.1, 'random_normal', 500],
 [0.5153661824172533, 2, 1, 0.01, 'random_normal', 50],
 [0.5158620501858483, 4, 2, 0.1, 'glorot_normal', 500],
 [0.5173610903100194, 3, 2, 0.1, 'random_normal', 500],
 [0.5175263795662178, 4, 1, 0.1, 'random_normal', 500],
 [0.5188486936158047, 2, 2, 0.1, 'random_normal', 500],
 [0.519013982872003, 4, 4, 0.01, 'random_normal', 50],
 [0.5209860171279971, 4, 4, 0.1, 'glorot_normal', 50],
 [0.5213822211935136, 2, 4, 0.1, 'random_normal', 50],
 [0.5223083311775838, 2, 1, 0.01, 'random_normal', 200],
 [0.5227295873214699, 2, 1, 0.001, 'glorot_normal', 500],
 [0.5233115035331599, 4, 1, 0.001, 'glorot_normal', 500],
 [0.5246972646942692, 3, 2, 0.001, 'random_normal', 50],
 [0.5249529592767572, 3, 2, 0.1, 'random_normal', 200],
 [0.5261214208885319, 3, 2, 0.01, 'random_normal', 200],
 [0.5266172886571269, 2, 1, 0.1, 'random_normal', 50],
 [0.5287546121693193, 3, 1, 0.1, 'random_normal', 50],
 [0.5307380832436994, 4, 2, 0.1, 'glorot_normal', 50],
 [0.5337132898552698, 2, 2, 0.1, 'random_normal', 200],
 [0.5374261712527401, 4, 1, 0.01, 'glorot_normal', 500],
 [0.5380108105164269, 2, 2, 0.1, 'random_normal', 50],
 [0.5390139828720029, 3, 1, 0.01, 'glorot_normal', 50],
 [0.5416471741527904, 3, 1, 0.01, 'glorot_normal', 500],
 [0.5426503465083665, 2, 1, 0.01, 'random_normal', 500],
 [0.5452949746075402, 2, 4, 0.1, 'random_normal', 200],
 [0.5459446948139475, 3, 8, 0.1, 'glorot_normal', 500],
 [0.5464519994009285, 4, 2, 0.01, 'glorot_normal', 200],
 [0.5507380832436995, 3, 2, 0.01, 'random_normal', 500],
 [0.5509033724998978, 2, 8, 0.1, 'random_normal', 500],
 [0.5529802442577642, 2, 1, 0.001, 'random_normal', 50],
 [0.5558734870042343, 2, 1, 0.01, 'glorot_normal', 200],
 [0.5561033125927539, 2, 2, 0.1, 'glorot_normal', 200],
 [0.560394297929119, 2, 2, 0.1, 'glorot_normal', 50],
 [0.5752008931610548, 2, 8, 0.1, 'glorot_normal', 50],
 [0.5794058300543249, 2, 4, 0.1, 'glorot_normal', 500],
 [0.5833667814886926, 3, 4, 0.01, 'random_normal', 50],
 [0.5857908423761351, 2, 2, 0.01, 'random_normal', 50],
 [0.5859561316323335, 2, 1, 0.001, 'glorot_normal', 50],
 [0.5887660489877055, 4, 8, 0.001, 'random_normal', 50],
 [0.5892619167563005, 2, 2, 0.1, 'glorot_normal', 500],
 [0.5919678135253992, 4, 4, 0.001, 'random_normal', 50],
 [0.5922973028169972, 4, 2, 0.001, 'glorot_normal', 50],
 [0.5979182267485397, 2, 8, 0.1, 'random_normal', 50],
 [0.5990763407627269, 2, 1, 0.01, 'glorot_normal', 500],
 [0.6029809250207633, 4, 2, 0.01, 'glorot_normal', 50],
 [0.603795934483369, 2, 4, 0.1, 'glorot_normal', 200],
 [0.603807371301755, 3, 4, 0.01, 'random_normal', 500],
 [0.6075050036080438, 2, 1, 0.001, 'random_normal', 500],
 [0.6105842308058873, 2, 8, 0.1, 'glorot_normal', 200],
 [0.6106427764238157, 2, 4, 0.1, 'glorot_normal', 50],
 [0.6114675888736095, 4, 8, 0.01, 'random_normal', 200],
 [0.6122256865494846, 3, 2, 0.01, 'glorot_normal', 50],
 [0.6124024126240691, 1, 1, 0.1, 'random_normal', 50],
 [0.620501586177788, 1, 1, 0.1, 'random_normal', 500],
 [0.6225463259220934, 4, 8, 0.01, 'random_normal', 500],
 [0.6225583073508789, 2, 1, 0.01, 'glorot_normal', 50],
 [0.6257908423761352, 4, 8, 0.01, 'random_normal', 50],
 [0.6264519994009287, 4, 8, 0.001, 'random_normal', 200],
 [0.6282587444007243, 3, 4, 0.01, 'random_normal', 200],
 [0.6344907212003212, 1, 1, 0.1, 'random_normal', 200],
 [0.6374223589799447, 2, 1, 0.001, 'random_normal', 200],
 [0.6386834043596062, 3, 2, 0.001, 'glorot_normal', 50],
 [0.6393331245660135, 2, 8, 0.1, 'random_normal', 200],
 [0.6423788582242977, 2, 2, 0.01, 'random_normal', 200],
 [0.642403365692268, 2, 1, 0.001, 'glorot_normal', 200],
 [0.6509733549362126, 1, 1, 0.1, 'glorot_normal', 500],
 [0.6521287458984029, 3, 4, 0.001, 'random_normal', 200],
 [0.6576916688224163, 3, 2, 0.01, 'glorot_normal', 200],
 [0.6578569580786144, 3, 2, 0.01, 'glorot_normal', 500],
 [0.6725562650618814, 1, 1, 0.1, 'glorot_normal', 200],
 [0.6750470407232426, 3, 2, 0.001, 'glorot_normal', 500],
 [0.6781875365910113, 3, 4, 0.001, 'random_normal', 50],
 [0.6792470761229186, 1, 2, 0.1, 'random_normal', 500],
 [0.6805601317957166, 1, 1, 0.1, 'glorot_normal', 50],
 [0.6820488243223004, 3, 4, 0.001, 'random_normal', 500],
 [0.6861859572208531, 4, 2, 0.01, 'glorot_normal', 500],
 [0.6869478671695236, 4, 8, 0.001, 'random_normal', 500],
 [0.6904807548300134, 2, 4, 0.01, 'random_normal', 50],
 [0.691895108037088, 3, 8, 0.01, 'random_normal', 200],
 [0.6961011341511567, 4, 4, 0.01, 'glorot_normal', 50],
 [0.6964268111699593, 3, 2, 0.001, 'glorot_normal', 200],
 [0.6975854697945456, 2, 2, 0.001, 'random_normal', 50],
 [0.7015551349953026, 2, 2, 0.01, 'glorot_normal', 50],
 [0.7031462142769617, 2, 4, 0.01, 'random_normal', 200],
 [0.7033727722030851, 4, 2, 0.001, 'glorot_normal', 200],
 [0.7035396952904815, 1, 2, 0.1, 'random_normal', 50],
 [0.7056873664002613, 2, 2, 0.01, 'random_normal', 500],
 [0.7145397361362615, 3, 8, 0.01, 'random_normal', 50],
 [0.7160978664887606, 2, 2, 0.001, 'random_normal', 500],
 [0.7167590235135538, 4, 2, 0.001, 'glorot_normal', 500],
 [0.7205563195229213, 2, 8, 0.1, 'glorot_normal', 500],
 [0.7212245564829061, 1, 2, 0.1, 'random_normal', 200],
 [0.7251914986316662, 2, 2, 0.01, 'glorot_normal', 500],
 [0.7253753046414421, 1, 1, 0.001, 'random_normal', 50],
 [0.7263517910874507, 1, 2, 0.1, 'glorot_normal', 500],
 [0.727209824771604, 1, 1, 0.01, 'random_normal', 200],
 [0.7276920772802157, 1, 2, 0.001, 'random_normal', 50],
 [0.7295298650727735, 1, 2, 0.01, 'glorot_normal', 200],
 [0.7311773115307637, 1, 4, 0.1, 'random_normal', 200],
 [0.7313006657862132, 2, 2, 0.001, 'glorot_normal', 200],
 [0.7326496657453675, 1, 1, 0.001, 'glorot_normal', 50],
 [0.7331215706563917, 1, 2, 0.01, 'glorot_normal', 500],
 [0.7331232044875896, 2, 2, 0.001, 'glorot_normal', 500],
 [0.7334494261167916, 2, 2, 0.01, 'glorot_normal', 200],
 [0.7369460971857256, 1, 1, 0.01, 'random_normal', 500],
 [0.738764823614307, 1, 1, 0.01, 'random_normal', 50],
 [0.7397309624627282, 1, 1, 0.01, 'glorot_normal', 50],
 [0.7397554699306983, 1, 1, 0.001, 'random_normal', 500],
 [0.7399033316541178, 1, 2, 0.001, 'glorot_normal', 200],
 [0.7405644886789112, 1, 4, 0.1, 'random_normal', 50],
 [0.7420531812054951, 1, 1, 0.01, 'glorot_normal', 200],
 [0.7427295873214699, 1, 4, 0.1, 'glorot_normal', 50],
 [0.7448652770024644, 1, 2, 0.001, 'glorot_normal', 50],
 [0.7453769384726401, 1, 1, 0.001, 'glorot_normal', 200],
 [0.7455215325336637, 1, 2, 0.01, 'random_normal', 200],
 [0.7455231663648618, 1, 1, 0.01, 'glorot_normal', 500],
 [0.7465127234604546, 3, 4, 0.01, 'glorot_normal', 50],
 [0.7470118588914455, 1, 2, 0.1, 'glorot_normal', 200],
 [0.7486712867782209, 1, 2, 0.1, 'glorot_normal', 50],
 [0.7494906531240214, 1, 1, 0.001, 'glorot_normal', 500],
 [0.7501512655384154, 1, 1, 0.001, 'random_normal', 200],
 [0.7501703269023923, 2, 2, 0.001, 'glorot_normal', 50],
 [0.7513121026045994, 1, 4, 0.001, 'random_normal', 50],
 [0.7513464130597574, 1, 4, 0.1, 'glorot_normal', 500],
 [0.7534526937791878, 1, 4, 0.001, 'random_normal', 200],
 [0.7541143954143804, 4, 4, 0.001, 'glorot_normal', 200],
 [0.7551099432243659, 1, 8, 0.1, 'random_normal', 50],
 [0.7561005895407573, 3, 8, 0.01, 'random_normal', 500],
 [0.7576121557041131, 1, 2, 0.01, 'random_normal', 50],
 [0.7587397715359374, 1, 2, 0.01, 'glorot_normal', 50],
 [0.7590774299835255, 1, 8, 0.1, 'random_normal', 200],
 [0.7592432638501233, 1, 4, 0.001, 'glorot_normal', 50],
 [0.7599038762645172, 1, 2, 0.001, 'random_normal', 200],
 [0.760063174806323, 2, 2, 0.001, 'random_normal', 200],
 [0.7600707993519136, 1, 8, 0.001, 'glorot_normal', 50],
 [0.7620488243223004, 2, 4, 0.01, 'random_normal', 500],
 [0.7645308862172724, 1, 8, 0.001, 'random_normal', 50],
 [0.7646950862526719, 1, 4, 0.1, 'random_normal', 500],
 [0.764858197067273, 4, 8, 0.001, 'glorot_normal', 50],
 [0.7656906340626574, 1, 8, 0.1, 'glorot_normal', 50],
 [0.7663496126458534, 2, 4, 0.001, 'glorot_normal', 50],
 [0.7665170803436492, 1, 2, 0.01, 'random_normal', 500],
 [0.7670113142810462, 1, 4, 0.1, 'glorot_normal', 200],
 [0.7678350375100413, 1, 8, 0.1, 'random_normal', 500],
 [0.767999237545441, 1, 4, 0.01, 'random_normal', 500],
 [0.7701479978760195, 1, 8, 0.01, 'glorot_normal', 50],
 [0.770313831742617, 2, 4, 0.001, 'random_normal', 500],
 [0.7704829332716108, 2, 8, 0.001, 'glorot_normal', 50],
 [0.7706476779174098, 2, 8, 0.01, 'random_normal', 50],
 [0.7708096995112121, 3, 8, 0.001, 'glorot_normal', 50],
 [0.7708118779528095, 1, 2, 0.001, 'glorot_normal', 500],
 [0.7714697673152069, 1, 4, 0.001, 'glorot_normal', 200],
 [0.7714746688088009, 2, 4, 0.001, 'random_normal', 50],
 [0.7716350565714052, 3, 4, 0.001, 'glorot_normal', 50],
 [0.7719634566422046, 2, 4, 0.01, 'glorot_normal', 50],
 [0.771967268915, 1, 4, 0.01, 'glorot_normal', 200],
 [0.7719694473565972, 1, 8, 0.1, 'glorot_normal', 200],
 [0.7721331027815976, 1, 8, 0.1, 'glorot_normal', 500],
 [0.7724647705147929, 1, 4, 0.01, 'random_normal', 50],
 [0.7727937151959916, 4, 4, 0.001, 'glorot_normal', 50],
 [0.7734603183247785, 3, 8, 0.01, 'glorot_normal', 200],
 [0.7739501953789808, 1, 4, 0.001, 'random_normal', 500],
 [0.774281863112176, 4, 8, 0.001, 'glorot_normal', 200],
 [0.7747788201015698, 4, 8, 0.01, 'glorot_normal', 50],
 [0.7751066755619698, 2, 4, 0.001, 'glorot_normal', 500],
 [0.7752725094285675, 2, 8, 0.001, 'random_normal', 50],
 [0.7767612019551513, 2, 8, 0.001, 'random_normal', 200],
 [0.776922134328155, 3, 8, 0.001, 'glorot_normal', 500],
 [0.7772565251133471, 2, 4, 0.01, 'glorot_normal', 200],
 [0.7774174574863507, 3, 8, 0.01, 'glorot_normal', 500],
 [0.7774190913175486, 1, 4, 0.01, 'glorot_normal', 50],
 [0.7774234482007434, 4, 4, 0.01, 'glorot_normal', 200],
 [0.7775832913529483, 2, 4, 0.001, 'glorot_normal', 200],
 [0.7780791591215435, 4, 8, 0.001, 'glorot_normal', 500],
 [0.7782455375985404, 2, 8, 0.001, 'glorot_normal', 500],
 [0.7782471714297384, 3, 8, 0.001, 'glorot_normal', 200],
 [0.7782493498713358, 3, 8, 0.01, 'glorot_normal', 50],
 [0.7784070145819435, 2, 4, 0.001, 'random_normal', 200],
 [0.7784146391275342, 1, 2, 0.001, 'random_normal', 500],
 [0.7784168175691314, 2, 8, 0.001, 'random_normal', 500],
 [0.7787457622503301, 4, 4, 0.001, 'glorot_normal', 500],
 [0.7795645839857313, 1, 8, 0.001, 'glorot_normal', 200],
 [0.7795678516481271, 1, 8, 0.01, 'random_normal', 50],
 [0.7795700300897246, 2, 4, 0.01, 'glorot_normal', 500],
 [0.7799000639917218, 2, 8, 0.01, 'glorot_normal', 50],
 [0.78022955328332, 1, 8, 0.01, 'glorot_normal', 200],
 [0.7802344547769139, 3, 4, 0.01, 'glorot_normal', 200],
 [0.780562310237314, 2, 8, 0.001, 'glorot_normal', 200],
 [0.7807226979999182, 1, 4, 0.01, 'random_normal', 200],
 [0.7812201995997113, 3, 4, 0.01, 'glorot_normal', 500],
 [0.7812207442101106, 3, 8, 0.001, 'random_normal', 50],
 [0.7812240118725067, 1, 8, 0.01, 'glorot_normal', 500],
 [0.7815485996705107, 3, 4, 0.001, 'glorot_normal', 200],
 [0.7815551349953027, 1, 8, 0.01, 'random_normal', 200],
 [0.7817133443163098, 2, 8, 0.01, 'glorot_normal', 500],
 [0.7823810366658951, 1, 4, 0.001, 'glorot_normal', 500],
 [0.7825463259220935, 2, 8, 0.01, 'glorot_normal', 200],
 [0.7828720029408962, 3, 4, 0.001, 'glorot_normal', 500],
 [0.7832004030116955, 4, 8, 0.01, 'glorot_normal', 500],
 [0.7833695045406891, 2, 8, 0.01, 'random_normal', 500],
 [0.7833716829822864, 4, 8, 0.01, 'glorot_normal', 200],
 [0.7836979046114887, 3, 8, 0.001, 'random_normal', 500],
 [0.7843645077402753, 3, 8, 0.001, 'random_normal', 200],
 [0.7851876863588709, 1, 8, 0.001, 'random_normal', 200],
 [0.785843397279671, 1, 8, 0.01, 'random_normal', 500],
 [0.7858515664356611, 4, 4, 0.01, 'glorot_normal', 500],
 [0.7865100004084579, 2, 8, 0.01, 'random_normal', 200],
 [0.7868422127520525, 1, 4, 0.01, 'glorot_normal', 500],
 [0.7876632129290508, 1, 8, 0.001, 'random_normal', 500],
 [0.7891535392868326, 1, 8, 0.001, 'glorot_normal', 500]]