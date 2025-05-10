import re
import random
import argparse
from typing import Dict, Any
import time
import json
from datetime import datetime, timedelta

from faker import Faker
from kafka import KafkaProducer
import numpy as np
from threading import Thread

from apm.data_generator.request_data import chooseFieldRandomValue, generate_random_string
from apm.tool import getISOFormatTime

appid_values=["pro-api-g10-xingyun","pro-api-g5-shentai3","scv-1","scv-2","scv-3","scv-4","scv-5","cif",
              "mdm","pro-api-g5-shentai1","pro-api-g6-shentai","cbpt-y","fmsbService","bqgps8","cbpt-y",
              "app-member-activity-xc","lf20uap-order","wechat-gxwx","lf20uap-order","pro-api-g8-chengbao",
              "club-point-account-xc","bwyl20_h5","ibox-claim","ibox-member","ibox-configs","ibox-common",
              "product_h5","bxhds","lf20uap-rule","huhuibao_web","tms","cpic-app-services","ibox-message",
              "ibox-report","ibox-upload","lf20uap-insure","pro-api-g7-shentai"]
appsysid_values=['bda14c5a-82cd-4087-8499-096b29b541c1', '086b7cac-202d-4cd1-8a33-917c6114d075', '3794a12d-6345-49a0-805f-d3268624eee6', '31f7e232-20d3-4fdb-8ab1-bbb8a5bbe7a3', '46ba40f3-12d3-4dac-af13-19510f004a8b', 'a15e7d7d-451f-4100-8f07-26abf4e607a4', 'b99af8ff-bfaa-4003-8316-273722d8f2f4', 'dc58a075-d6d1-4c8a-91ea-b0bc57ed4a9a', '0051d2db-efb7-46c8-9abd-165b04c1f886', 'f09b3573-9b59-4480-8c18-4f758d521d53', 'cc02b5ae-f420-4c2c-871d-032c25ffdc77', '7b0a939f-6c8c-460b-a4e9-a8c5b0d94448', 'bbc09e2d-0f5f-47ae-8754-04201d6bff77', 'b7fdd9d2-3be7-414d-8919-fa68f25c77d6', 'e1323678-ef5a-4389-9a19-d6d5b2b57c9a', 'a17cc966-70c8-4d30-9ef1-8ea065f838a1', '2bcff569-a7ec-410c-ae07-50a90fe74594', 'bf763e94-85d7-4317-aedb-ff3307d69336', 'ead1fb0d-8cae-4048-87a7-d3c80fc9c207', '2b41bff4-90be-4c08-af0f-3b75332c4be7', '3106bf25-7b57-435f-8fe4-4d3f879cb163', 'da10a00c-4438-4c90-8023-45fe3c45d851', 'f96d2e3d-37fc-4cea-9a28-e0cf15365092', '9b4a432e-5967-455f-8114-98374f870dae', '1070bcd8-0b2d-4f40-8ec8-98106a1b9681', '6b999963-2c75-416e-9aa0-8493fafd3b9b', '945f5913-7371-4f30-ac6a-be4835e98175', '93326905-6f64-40bb-8bf8-16133981af34', '49669e44-547d-423d-90e6-b0cda263b3bc', 'e79e4156-418c-4116-8031-2ec78ece68bc', '565fb6e3-c240-4688-ae6a-59a9ae940b2c', 'Other', '0ce9bd9a-f197-4ca2-8180-fe667e368ba7', 'bcc6a3d0-62ce-44ec-9510-4f41ec4e16d0', '90e9f967-3eb7-4acb-bebd-97b74b491036', '96050743-338e-44d0-ad23-5cad80b92170', '6b3bf5c3-c197-4274-83f1-ec84c6ab0065', '64d18050-da9b-40b5-926b-c7677616ba4b', '1c27460b-c32f-4c6a-86d3-4ca75c366177', '584e7de8-ceca-4012-85a9-e4a4048c1394', '6b23c661-6905-42f3-ae35-48fece9eb69c', '27aca7be-b81e-4b05-9966-06666c07f1cf', '5352ea3f-3469-41b9-a74f-6cedc4fbfa97', '64c311d9-2109-4d8c-8455-3bbdaa8a7415', '2ee0b8a5-8de5-4742-8506-f72850a9b287', '1638084c-89c9-4341-859b-4032411c2cea', 'f6a15095-db42-4132-b458-b199c4bf8cd9', '0afd7d4a-6be9-4d33-8f7f-c21a534abe49', 'a7b547fa-c8ea-40e4-a02d-fe672b72ffcb', '82174ede-48d9-4a61-8050-fe830111e886', '6c9d8654-85de-4b8b-8cdb-128c888b8e64', '2d2c5d75-0079-4e66-9db6-c6f42fc3b333', '21198929-a19f-42a7-8ba1-478d1af41e82', '6cdca5d2-ff63-4479-b05e-46c758c9ff79', 'fb00bade-a826-411c-8bf3-aba25be36239', '64520f6c-8bda-486b-a541-f43f366ff236', 'ca5fbf7d-7b32-40ba-97de-91ac67084191', '03e99a87-40f6-4395-85ee-fb1db644cb72', 'b82017e3-0187-440b-84f0-6b66c6ae067a', '9c2e1b41-b326-44f2-b9a9-f6e7c62e6c25', '8705098d-7a92-4518-93ec-169fefc65d52', '64f76adf-3388-4eb3-a912-f3b476c7654c', 'd42df83a-35ed-412e-9918-8730f0db87ca', '3c98113e-1734-40ae-812a-5ae3ffcdd39f', 'ad6edd3d-d9ea-4341-8dce-30c6c927b56c', '5af9ab36-1b31-4868-8965-a2fb16811ccc', '90097ef9-5a4f-47d8-86b3-0c8612e73e1e', '6fea236d-3bc6-4acc-8f87-af1f531da761', '5506750f-491e-4e87-96f4-856e6e7c9443', '89c9d48c-482c-4f44-8120-48c2e414dd7c', 'efad028b-042e-40a8-b0d2-7638f4f57be3', 'fa5b2e6c-f973-4a2c-8461-fab7a4f209f2', '95a3ed77-6fa2-4137-9458-d204a33f52f4', '331b7e39-c687-432c-8f00-ee2a5ed47d2b', 'ffe66d6d-cd78-4cb7-9337-38b54d34bdd6', '6a9749c9-9aac-4e52-95e0-d94ca8a05616', '75055c32-bbec-40d0-8075-3fb1e57671b9', '6c55c869-c2ae-43ab-89a4-d2955f677539', '9ba9403b-b000-4a4e-9d85-8e831bbf9d06', 'ba2be26c-ab81-41a9-96b5-ccc6fc449074', '5ce7a9a2-a10e-4a70-a0b1-a8e178b98f14']
group_values = ['E01090DB3A6CC1BA', '112B025F88838E9F', '3DF9D4ECFB6B1791', '40CA8BDA95BD41A3', '628747FC25DAB8A8', '96299FC7A384B583', '9926CA6C668D6DF8', 'A7C080C28F86A01B', 'B29538867F5098B0', 'B3D6282393299C8E', 'E820303E475A700D', 'BE24E42E554C0C52', 'b3945a3c74b4715d7c02ef2615dabc70149179d6', '0589D845BFD45875', 'ee5a65608447ea8b36b7587637c26f2858cd74de', 'ca0bf7c3505686fa1b9c5c4619e3138797910f33', '9ac5f8d7caba37113e0c9eefa3251c76e747ced6', 'a8ab4e67b2590a322fbb5bf723528f226f5b07da', '9439c7e254481dd911789e9664bec50326815baf', '8e276425b5ecd973353c95ed7b38b17d6890de03', '205fc7bbfe19e03b1bd95ed0b7fe40e94677412b', 'facea6791049b34888c49467e0e9235e68affd1e', '615195628b1b0d3d111947e28aa4245b20e8e9a9', 'ae8726725676353cdf49fe0d6c6b8f43563c49ca', '1555EF2E8B588ECF', '209313B94AA713EC', '28C5959502A8DDFE', '2C319103ACBD0DAD', '55B71EAF8A8189F3', '595D5DD7437FAA65', '6275DFDB20FD95C1', '7608049CE40C959D', '8417E1D37887FCC6', '8661E9E02068E690', 'A0D726C11754ACF8', 'B289508BDFBC3E8A', 'C64A2103C0ECD73C', 'CDA2961469FACEAE', 'D7E4142232B7ECBF', 'FAB145D8EE1EB13B', '8AEB79A4E4D6BA1B', '5394A997CDDEDE6F', '3b66a23e83b95bbe7d349223d7db80e72b72635a', 'aabea7b9eded8275bb73be6bd3de6825735afa5b', '12b41ab91459a4e8a1226a71ce72a38b9d485e62', '1cf8c37b786c4b15b2bb1c377d8937ca647abea3', '55A4AF49469CE198', '89a7f8f20a9c69f2795848b49dfcabb8df977701', 'd93e3be1d3210c152841696a3b8cac6d4e1c341b', 'ddaddab029c9b885b3d9a1b7f9eb0ea0520544e3', 'ca4e7b43ac26bb84f385b45a7250c50b9051d4e1', '24c37d6aa514bd0af08edd54c0a5922abffe7fab', '22278f35ceeb62fe88df68811b67fa85a699a6ee', '0AEB5CA681AB72FE', '196174329C13354F', '22A0E554756A0FC4', '4802E0642B920D7D', '539360420134E83D', '73E4A63816203C05', '954C9EB945966DAD', 'A088FCD006F869AE', 'A1F6842045E07718', 'B01A343B951030CE', 'B30206BB0F0B1E8B', 'CF092DDB2607B399', 'D091E7665A3E9D9E', 'DA91E840391272EE', 'E893E6F2BCEC4EC5', '7b9e7c0aa51899834a8291c868137e05865ef1ec', 'e4864a2d38da57268729d358d1cf8d32277d9b72', 'ff151ce3f5991142f84eb4d93fbd22095a0a1635', '01EC5C2295C84F98', '0B8A2384B7CD4DA9', '15BB92C821EB0921', '17CCBDC08F4F2C00', '1A5F228BFBFAC376', '215A400D72A9B472', '23E5ED466F857EA2', '2AF6B484C40775A5', '39FEE6FE2217C5DE', '45ABD1A3FE34372E', '58F7B1E88A1EC619', '590FB1C6E617AE56', '5C0576B238EEB14A', '8ECC951AE9B2C3AC', 'A0936DE8AB291C67', 'B4F98A19E7D2C3F3', 'C31A76DE7D075801', 'DF3648FC3C06042F', 'E25D16E60D2C7539', 'EECE0B99B080C335', 'F53653D48C58AB3B', 'F6A5008649B882D7', '9686f8b025fd6d780136362641d1f8e41c7dcdfa', '1d821a1e974a127fd7ad58b983b4cff2b5b936e8', 'ab9794e166b96ae7c522870d6c16241d7bc3ff3f', '6b20ece6eed591ad78b0c46d914ba48bec9b3308', '8d5a1208937a408de911ec0ed6c657a78295cbd9', '8c21e8d11e777bc5c890f0b48c452327a83be536', 'e8168b449ecc4003f269f72f6f961d8f8a91573a', '4522d179d6531b57a7b89d797e5b9b316619f9f6', '7e3094852e170c061052efb0b32963901bf62cde', 'c3638281e481637e0ce65be3da5f04a0d52e0d2c', 'd84ed786c9a4bedd7abf78ee4a8be1acd519e7a2', '928c8c0f5a90419ac9c9f872655d65fc9b8125ba', 'c8a128ac61b5d421f4148780adcf0457037cd1be', '1ef561048d8966d351109b04b7bdba979b9a24d0', '251660401921868ce2ba065c13893cb173740236', '6a138b6b8fec66c596de2ce87eb07aabef9663ec', '13b0a45ef246be33a48d1d586362036be7d2110c', '18948253e29c2d86349f383c2997a52532e83ee4', '1e5d07413756f1d595cee10378ae0262579a3639', 'c7f8fdabb1e336c4cfc149c68e5bb9ea4397a659', 'e0e246f2db8921991e302009fed352101a5af364', 'a5f84eb984f19df2d286d86c73cc29ae0ccb3a91', '51df6106f03af244e94ee4752b0054f429a9327f', '494f3b046b0b7b1e66bac36177c1a72452a139ca', '0715CFBBAB1AE619', '123FE1080581822F', '29C058E812D46BFA', '2A410669C7688E46', '2FAE2B45A03E36FF', '38B295BD67191082', '3CF67BBE6E59D960', '4728B6B3F483F708', '4D459968A47F8DED', '5CF39F16AEF51754', '5FE9D566A5CE6DCF', '63AE755212BA3E29', '81FB0B18362520B7', 'A3E6B00A248F1BE8', 'AA7A1FD9B545C91C', 'B3AA64A4F836AE89', 'BBF9B0D643274EC2', 'C34E0F6FE27DD0FE', 'CC0F28B9A293BE80', 'D2EA9EE8B20FB933', 'D355B6EC6311B71A', 'DDA572D317A29C4A', 'E048C01CF2E6DFB4', 'E292A92787705EE8', 'EA4BD1550A014C53', 'EC7B69F0FBE99585', 'F666BE309BA7B05F', '1e1f688da03c08ebf0bf4c7cb6218b821b49b712', 'a11b560cdd71b22d257b4495b6cf0be3d7314840', 'ea83a8c6a970d07f1f5e58722446bda31591d89a', '5ccff7542987e349befa5a27d947c0c080116412', '4fc02577955487e29a93c0c888989f4bc7baccc6', '919037ecc1051122fb94e697dcb1ad34012b6464', '18c6d5043fe15263aa4ea1a3a2c896bbd99f9b82', 'bb6ab84c4053cd77cce3b5e30a431d11bd6c8e46', 'f787ac8d7d78468b497a7dd9e8d86697a4645b9f', '051795f61ee3f1db110d6b94205f903f95e25614', 'bba42a6f8eb266c46b42a985ca4eefabeb746380', '516c381cf6adfa4cf3fc32cc4504ebf095675b69', '173afbc91b08c19d3bdd595ee8d14149a357a67f', '41e0948c77fb657d2df6c70e8c38f9d4fe893bc2', '865324044b3b002875cd337604ea6256b466a478', '16af10cc26f44c9e31b105d72f710ce08942ca0e', '37e1fe23c26b2ee50d42dec15816002f9890340c', '1f986e7b718c204569d3cf6c1050aeaca87566a4', '540175cef518472e2c2aa7008a0c2726dbe44fdf', 'c3a299648b12e093dd1544492bb6f2318132a2c2', '297721f8a567296b2a6f73adeacb156fa4f9fcfb', 'e44a32a870ea9d3f3b3599e89d0dbaed4fca2677', '253041ad22e655fe4c54e4695d023a1ec017dc0f', '93010c268aa2e6b0292ccc00ccc083d798ebe543', 'a2107abe419713d2862ef1f448622184ef02622f', 'dba0f5311a39180626cc67be4761c38377a0ef57', '141756948e1f8ed0d71183f454216a184ecee2d4', '96ed6c3b93010d316e8feded93bfa23e633ed171', 'be0749144f6e603b43654c87c2dbd9efdb765cb5', 'f4b5bdb5b6e32ace014a910f1c149d12647459e3', '84c1fe0d7a411349c10ff7153668c1f075f7c863', 'f4f060881f27025d78c9201411b0d7522194c44b', '299883b0592a9e7d8d6dccb67342e32b59e484eb', '793a6f9b7926d1efcfe0776b3ff7346b3e9ec8d9', '0e301faa0bd2b0e2cfc44bca88455b2fea37bbd6', 'eadd2099677a0395fa60f7a6fd4adb60da7ebfdc', 'a9e89fc580f22d60cbb5d5f8767b7df0a5d16c53', 'bef3cf0d7d7fdb32b3a5adfe99542a556fd1b91e', '41bab090fe65fc5436dcb84da7f8237f4b1d5b7d', '0c9854f16e398c75ba73545658808f9b06efbf07', '6e68c012c88423b80c5eac485db61853e475d754', 'c05c0af311a48f5e845e61f489cff3aebaae258d', '4b49fc7315d43f4d07b4cf595dca9cbde480b5d8', '39b9655790c1c95f300640ecda19df66575d7078', '8e6d25d652c730f3e8d222bcca66422f2ad4b424', '62716a1349c75d739a69e5cc26764823d9670342', '271755c5946e6b39017eba015aaef924f1058641', '920a53c8caa3ad92e99bcacc77cd08369df114ac', '9530e963092f309f93e89554c818066e92aec207', '1ac1bbec80f99e65b85a3747da7bef34c16da908', 'a31f993ee971f0253d5fbaca2410be9e5c11a4c1', '67a316f4d6f38c88c94132843f8b1e95cb8e237a', '96a67ce5caacd7ed7558bd3f52e457a687c19d76', 'fa4dc858f7b2b63cdbed39cb64d8de2a19218166', '105ba23347e1756c819cd2504d399ec11750fa51', 'b8f637d0421bdcfd2d769a2eb6450250838559e9', 'af54f202b7fc114fb6f705d71e45f41352407ca1', 'aa26a99021092cb2eb6f67e2dc9b0f3431a73f12', '16f32505ceb98644137ef2cb74cb82c1b3bb6897', '16da9c945433ce52128145d283d3fc3a9d7f1428', '13695f5a0009fef2a69037621442fdcabac6c746', 'bd83b7ec48e9ac2628c94ecc3a009ce0c8870848', 'bb3bd559fffada2c2705c181748fccca72aff852', 'e34fae7eaed3dada0ab48506c4c9627c59f640ec', 'db0449a4e2839c4eb86726b3ac2ab38302b7a144', 'a017816aa7f1a4b0555087d6b09081fb134a6124', '9c1299c409cfb9cb631cae9dae956b92cb34d68f', 'b42f4b943d5b351a8883dd43d163eb2701f1a63f', 'b815aad58d30f9802581f8f9b674f6a4f921285a', '7ac73b527e4af1e8f5ff0c5e197194db127949f7', 'b092581d80ae871f4580af2e19b8f497cf10b825', '4a0511cab794a7b695edeeae1517d560cd76de6f', '94ea8fbbd2056b2b1fe6aecada74c9b92ad0e0a3', '908f9a88bc3f3ebdd0173f4bd110da918e42b1ee', '0f259ef520e6d8f797571fd50055166e53aa9509', 'dcf1c5213d09f6362b27f9a854ed9d1a70408024', '2ff8bb04d485d7ea40183f01480809aeebb60f62', '1994cc60a5f035f91547a3a25f0b7c65d864e8d3', '18b2b4477fb7a0c5b4d367c4c7697ecf717c8efa', 'e324c3c22304781502d08ea7c4699fd9201cccda', '939760ec67941383f81a891da5bac41fa84df4e7', '20bf068ff24a56341f655d09bd7490a5f5ff1f0d', '5020933e2f878bc730ca8f97a6b792311a02add6', '2518104917487471cc7bd9e072c33087c7404c36', 'ae7a2dd0f01884bd4d46f1078c8ade322b09d22e', 'fa3fbe8a9e56264d4f1edca9e49d6fbf1d22ed22', 'f239927278bae324ba21738377cad3f498b15c93', '11275fe5707e4ebc535602d1a4b92b01631cd1eb', '8eed9df3d09ef1628882ee09db4427654c408280', '73d3bc1253996d2368b4529cc2f5b71c2b5f01ac', 'e4e83beb777952ca228076aa20c9187b0ec4e96f', '25bca5ebeed727943f9402f374594456bec46390', '616d471bd202659da96f23cf3863d8e18688eda1', 'D31A3BD81465E5ED', '24edc66027f53e0bcc6edf07e2a9dad4fab3ed93', '45f2705e0c254918f54a6ff4ab2a81ad0d576050', 'dc555b10ec9cb6e314e579a36d4b248a48d85b8a', 'd7095b8430465aa2feaffa0e76a7a4cfe9ae0b8a', '71e16014d93a9a4a1ef1738dac71d619d3ebda6b', '954348c3cc6face74bec465b36c0f5ac66745641', 'c60d536ca758ec8ba95675e04c2bbbc67db06b6f', '1c12769e4d0b2ee4d774b98316b205936d5a4e3e', '084054ee7c8e6561008b0fec67a7da7d6c8cd679', '4ee7094d04536aa8b3625ff56e24f183b143601f', '74181b87f9fd3f1322734ccb687268017ca209db', '7224e6f78e7ef076eb2d5125abe647cd86aab759', 'b3faf6a23bf619f3e49dbfabcc44028bd9de6495', '8bd9c4d9dc262f5263d2716c24424323191cf3d2', 'c8500023df98d47ef731bf6fcfe7c311db433987', 'b82e45f6d9c9114a549592bb5fb0143282280026', '162c922784847d2159a0b008fc782e7a4a82c563', '82e59bcdfd23eee8eb1057a33723cc8dbbac4dad', '83299467094f1be75537ea658bcc6d5f99cee447', '0333515b10e6719b19df5c8e4956889fd3b617b0', 'f7d8a0ffb50dd80a7accd235ebc4a6baf0f9ae6f', '3d8a72572b35a40d7227874667ee7fde87e0e732', '28aa04a6a9e83f5eaa8d93311a5d907f04292d9f', '4697564a8282b5213e6494777f6ffff1e4df5e10', 'd19cd583cb4970ccb188795e93639c41da75ae15', '1b4a32e3bfeaa58fa2453670f5bf61c9e189eea0', '6b14e9d04201e17fc7ffca0ce9954b04ecf71d63', '9d6e76cca23098195d35b8ad04ec2407e73fdc2d', 'f7e2f41a22e406d2a221ecf5250d1a3b4b9cb2f4', '30043da0ec8e9a01ffc4292d499bbf66fc8f3fbb', 'ae2de8f05cedeaf65dbe76a32464ebce09335df7', 'b4ff40f1fffadaa525cad4de24ca5b631e517a52', '7bb26532a97b714e75a4053ff00739782c707345', '66e86a4626b49e8c9da70daa37a5134c128686fa', '5f2afc3ff9cb3bd1727fadd45fbdb742eaebbef9', '79961c4b2e2baf00a7209235efc8e039defb31d3', '213e440b4a018c6e62f4f3004e3c00a0b7925dc3', '79ba449e7d3cb25b0ae9e1ecd3729355529354cb', '9aa41c8f1e5a1e6496935e2d7098f6d9c9452e92', '05cb3dd00635589e6ac1f608040289798c4688e7', 'b59e45d10ff253abaa8f40c34117e3e4eb2a2a1a', '9ad5b6c15ee4cfedaba630e26941a32ebd0f7ccd', '617d1860702a49965dfd7e703a008935dc167bfa', '31e131f7eb0f4c5b3337ff67ee20e25607a11bc2', '9450156aef1d62f67a60ec35556b5e638bd7b9cf', 'dd71db7674ec6af044fba2bda425e52667e482b5', 'b1691a0878b8b2f54d777dca37e2364806b75a69']

def get_field_all_values(cardinality,prefix,val_lenth=20):
    if prefix=='appid_':
        return appid_values
    elif prefix=='appsysid_':
        return appsysid_values
    elif prefix=='status':
        status_values=[]
        for i in range(cardinality):
            status_values.append(str(i))
        return status_values
    elif prefix=='group':
        return group_values
    return [generate_random_string(val_lenth, prefix) for i in range(cardinality)]
def parse_log_line(line: str,i,max_rows,min_time,max_time) -> Dict[str, Any]:
    """解析单行日志，提取指定字段并用随机值填充缺失字段。"""
    pattern = re.compile(
        r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "([^"]*)" "([^"]*)" "([^"]*)"$'
    )
    match = pattern.match(line.strip())
    if not match:
        return {}

    # 提取已知字段
    extracted = {
        'ts': getISOFormatTime(match.group(2),i,max_rows,min_time=min_time,max_time=max_time),
        'method': match.group(3),
        'path': match.group(4),
        'status_code': int(match.group(6)),
        'dur': int(match.group(7)),
        'agent': match.group(9),
        'ip_addr': match.group(1),
    }

    # 所有需要处理的字段列表
    all_fields = [
        'ts', 'type', 'group', 'appid', 'appsysid', 'agent', 'service_type',
        'path', 'method', 'root_appid', 'pappid', 'pappsysid', 'papp_type',
        'pagent', 'pagent_ip', 'uevent_model', 'uevent_id', 'user_id',
        'session_id', 'host', 'ip_addr', 'province', 'city', 'page_id',
        'page_group', 'status', 'err_4xx', 'err_5xx', 'status_code', 'tag',
        'code', 'is_model', 'exception', 'biz', 'fail', 'httperr', 'neterr',
        'err', 'tolerated', 'frustrated', 'dur'
    ]
    appid_values=get_field_all_values(50,'appid_')
    appsysid_values = get_field_all_values(100, 'appsysid_')
    status_values = get_field_all_values(100, 'status')
    group_values = get_field_all_values(100, 'group')
    result = {}
    for field in all_fields:
        if field in extracted:
            result[field] = extracted[field]
        else:
            condition=-1
            if extracted['status_code']!=200:
                condition=extracted['status_code']
            # 生成基于基数的随机整数
            if field=='appid':
                result[field] = appid_values[random.randint(0, len(appid_values) - 1)]
            elif field=='appsysid':
                result[field] = appsysid_values[random.randint(0, len(appsysid_values) - 1)]
            elif field=='status':
                result[field] = status_values[random.randint(0, len(status_values) - 1)]
            elif field=='group':
                result[field] = group_values[random.randint(0, len(group_values) - 1)]
            elif field=='is_model':
                result[field] = random.randint(0, 1)
            else:
                result[field] = chooseFieldRandomValue(field,condition=condition)

    return result


def main():
    """主函数，处理命令行参数并解析日志文件。"""
    parser = argparse.ArgumentParser(description='解析日志文件并提取字段，缺失字段用随机值填充。')
    # parser.add_argument('input_file', help='日志文件路径')
    # parser.add_argument('--base', type=int, default=100, help='随机值的基数（默认：100）')
    args = parser.parse_args()
    args.input_file = "/data/lxhdata/access.log"
    # Kafka配置
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    max_rows = 70_000_000
    kafka_topic="dwm_request_src"
    min_time = "2023-05-17T00:00:00+0000"
    max_time = "2023-05-19T23:59:59+0000"
    total,rate=0,0.001
    while(total<max_rows):
        with open(args.input_file, 'r') as file:
            for line in file:
                record = parse_log_line(line,total,max_rows,min_time,max_time);
                producer.send(kafka_topic, record)
                print("发送数据:", record)
                total += 1
                # print("发送数据:", record)
                if total%1000==0:
                    time.sleep(rate)  # 控制速率sec
    print("发送数据完成", total)

if __name__ == '__main__':
    main()