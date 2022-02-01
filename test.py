import unittest
import yaml
from yaml.loader import SafeLoader

from nmt_worker.translator import Translator
from nmt_worker.schemas import Request
from nmt_worker.validation_html_xml import validate

with open('models/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=SafeLoader)

modular_model = Translator(
    # TODO: copy main method stuff here
    )


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

    # failed sentences:
    tagged_xml_sources = [
       '<g id="1">Yearly load - <x id="2"/></g><g id="3">Uncertainty: N/A</g>',
       # '<g id="1">Söön</g> <g id="2">ma</g>',
       '<g id="1"><g id="2"></g>YouTube</g>',
       #'<g id="1">2.19 Treatment and disposal of hazardous waste</g>',
       #'<g id="1">God save us</g> <g id="2">is the term I use for this project<x id="1"/>. <bx id="1"/>She might be a merciful <g id="4">woman. Vishna</g> help us.</g>',
       # '<g id="1">Orbital  subsystems and earlier development work</g><g id="2"><g id="3">[</g><g id="4">edit</g><g id="5">]</g></g>',
       # '<g id="1">Ewout van Ginneken</g> <g id="2">is the coordinator of the Berlin Hub of the European Observatory on Health Systems and Policies, He is a series editor of the</g>',
       # '<g id="1">Table 6. Organotin substances in Estonian landfill and storm waters (data from COHIBA WP3).</g>',
       # '<g id="1">The data for these industries is confidential and we cannot be sure if they use or emit Mercury or not. <g id="2">Further research is needed. </g></g>',
       # '<g id="1">IAOQ appoints a person responsible for the performance of the listed tasks.</g>',
       # '<g id="1">QUALIFICATIONS AND PROFESSIONAL REQUIREMENTS OF EMPLOYEE</g>',
       # '<g id="1">Views</g>',
       # 'Use MathJax to format equations. <g id="1">MathJax reference</g>.',
       # ', and the infinite places. This correction takes up a substantial portion of the <x id="1"/>HAT paper. That is, the isomorphism is generically true over <bx id="2"/>',
       # '<g id="1">Estonia has continued to align its <x id="2"/>legislation in this area.</g>',
       # '<g id="1">Yearly load – <x id="2"/></g><g id="3">Uncertainty: N/A</g>',
       # '<g id="1">Yearly load - <x id="2"/></g><g id="3">Uncertainty: N/A</g>',
       # '<bx id="1"/>As regards <g id="2">exports</g> of <g id="3">radioactive waste</g> from the Community to third countries, six Member States issued a total of 13 authorisations, representing 35 shipments.',
       # '<g id="1"><g id="2">Random url</g> statistics of important event 2009</g>',
    ]

    tagged_html_sources = [
        'Hillier, Tim (1998). <i1>Sourcebook on Public International Law</i1> (First ed.). London &amp; Sydney: Cavendish Publishing. p. 207. <a2>ISBN</a2> <a3><bdi4>1-85941-050-2</bdi4></a3>.',
        '<a1>Signers</a1> of the <a2>United States Declaration of Independence</a2>',
        'Boyd, Julian P., ed. <i1>The Papers of Thomas Jefferson</i1>, vol. 1. Princeton University Press, 1950.',
        '<div>I want this text to have both <b>bold <i>and</b> italic</i></div>',
        'email address<span1><span2> * <span3> Required</span3></span2></span1>',
        '<strong1>Signature is not valid </strong1><em2>- </em2>marked with red. This means that the digital signature has been declared invalid.',
        '<a1>"Did You Know ... Independence Day Should Actually Be July 2?"</a1> (Press release). National Archives and Records Administration. June 1, 2005. <a2>Archived</a2> from the original on June 26, 2012<span3>. Retrieved <span4>July 4,</span4> 2012</span3>.',
        '<a1>You are <a2> so blaa <a3> but I am </a1> very much </a2> aligned to gla. </a3>',
        """<a1>"A Closer Look at Jefferson's Declaration"</a1>. <i2><a3>New York Public Library</a3></i2><span4>. Retrieved <span5>July 6,</span5> 2020</span4>.""",
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
        request = Request(self.tagged_xml_sources, "et", "de", "general", "document")
        response = modular_model.process_request(request)
        tagged_hyps = response.translation
        self.assertEqual(len(tagged_hyps), len(self.tagged_xml_sources))
        for sidx, sent in enumerate(tagged_hyps):
            bool = validate(self.tagged_xml_sources[sidx], sent, "XML")
            print(f"sent:{sidx}\nsrc:{self.tagged_xml_sources[sidx]}\ntgt:{sent}\n{bool}\n")
            #self.assertTrue(bool)

    def test_translate_web_tagged_input(self):
        return True
        """
        request = Request(self.tagged_html_sources, "en", "et", "general", "web")
        with open(f"TranslationSource.txt") as fin, open("TranslationTarget.txt", "w") as fin_tgt:
            for line in fin:
                line = line.strip()
                request = Request(line, "en", "et", "general", "web")
                response = modular_model.process_request(request)
                print(response.translation)
                fin_tgt.write(f"{response.translation}\n")
        """
        request = Request(self.tagged_html_sources, "en", "et", "general", "web")
        response = modular_model.process_request(request)
        tagged_hyps = response.translation
        self.assertEqual(len(tagged_hyps), len(self.tagged_html_sources))
        with open("sources.txt", "w") as fin_src, open("target.txt", "w") as fin_tgt:
            for sidx, sent in enumerate(tagged_hyps):
                bool = validate(self.tagged_html_sources[sidx], sent, "HTML")
                print(f"sent:{sidx}\nsrc:{self.tagged_html_sources[sidx]}\ntgt:{sent}\n{bool}\n")
                fin_src.write(f"{self.tagged_html_sources[sidx]}\n")
                fin_tgt.write(f"{sent}\n")
                #self.assertTrue(bool)


if __name__ == '__main__':
    unittest.main()
