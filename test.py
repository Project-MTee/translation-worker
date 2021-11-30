import unittest
import yaml
from yaml.loader import SafeLoader

from nmt_worker.translator import Translator
from nmt_worker.utils import Request

with open('config/modular_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=SafeLoader)
modular_model = Translator(**config['parameters'])


class TranslatorWithAlignmentsTest(unittest.TestCase):
    srcs = [
        "Eile sadas lund.",
        "Seadusesarnased üldistused on kinnitamiseks sobilikud, mitteseadusesarnased aga mitte.",
        "Mu lapsepõlvekodu on siinkandis.",
    ]
    hyps_verified = ['It was snowing last night.',
                     'Law-like generalisations are suitable for approval, but not law-like ones.',
                     'My childhood home is around here.'
                     ]
    
    tagged_sources = [
        '<g id="1">2.19 Treatment and disposal of hazardous waste</g>',
        '<g id="1">Ewout van Ginneken </g><g id="2">is the coordinator of the Berlin Hub of the European Observatory on Health Systems and Policies. He is a series editor of the</g>',
        '<g id="1">Table 6. Organotin substances in Estonian landfill and storm waters (data from COHIBA WP3).</g>',
        '<g id="1">The data for these industries is confidential and we cannot be sure if they use or emit Mercury or not. <g id="2">Further research is needed. </g></g>',
        '<g id="1">IAOQ appoints a person responsible for the performance of the listed tasks.</g>',
        '<g id="1">QUALIFICATIONS AND PROFESSIONAL REQUIREMENTS OF EMPLOYEE</g>',
    ]

    def test_modular_translation(self):
        """
        Check that Modular model can translate
        """
        hyps = modular_model.translate(self.srcs, "et", "en")
        self.assertEqual(hyps, self.hyps_verified)

    def test_modular_translation_with_alignment(self):
        """
        Check that Modular model can translate with alignments
        """
        hyps, alignments = modular_model.translate_align(self.srcs, "et", "en")
        # Translation tests
        self.assertEqual(hyps, self.hyps_verified)

        # Alignment tests
        self.assertEqual(len(alignments), len(self.srcs))
        self.assertTrue(all([len(j) > 1 for j in alignments]))

    def test_translate_doc_tagged_input(self):
        request = Request(self.tagged_sources, "en", "et", "general", "doc")
        response = modular_model.process_request(request)
        tagged_hyps = response.translation
        self.assertEqual(len(tagged_hyps), len(self.tagged_sources))
        

if __name__ == '__main__':
    unittest.main()
