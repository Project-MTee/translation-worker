from html.parser import HTMLParser
import xml.etree.ElementTree as ET


class MTeeHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tag_stack = []
        self.tag_list = []

    def handle_starttag(self, tag, attrs):
        self.tag_stack.append(tag)
        if len(attrs) > 0:
            attr_dict = dict(attrs)
            self.tag_list.append((tag, attr_dict['id']))
        else:
            self.tag_list.append((tag, ''))

    def handle_endtag(self, tag):
        self.tag_stack.pop()
        self.tag_list.append((tag, ''))

    def handle_startendtag(self, tag, attrs):
        if len(attrs) > 0:
            attr_dict = dict(attrs)
            self.tag_list.append((tag, attr_dict['id']))
        else:
            self.tag_list.append((tag, ''))

    def isHtml(self):
        return len(self.tag_stack) == 0


def validateHTML(source, translation):
    if source != '' and translation != '':
        srcparser = MTeeHTMLParser()
        srcparser.feed('<html>' + source + '</html>')

        translationparser = MTeeHTMLParser()
        translationparser.feed('<html>' + translation + '</html>')

        separately_html = srcparser.isHtml() and translationparser.isHtml()
        srcparser.tag_list.sort()
        translationparser.tag_list.sort()

        return separately_html and (srcparser.tag_list == translationparser.tag_list)
    else:
        return False


def validateXML(source, translation):
    if source != '' and translation != '':
        try:
            root1 = ET.fromstring('<xml>' + source + '</xml>')
        except ET.ParseError:
            return False
        try:
            root2 = ET.fromstring('<xml>' + translation + '</xml>')
        except ET.ParseError:
            return False

        tags1 = [(elem.tag, elem.attrib.get('id')) for elem in root1.iter()]
        tags2 = [(elem.tag, elem.attrib.get('id')) for elem in root2.iter()]

        tags1.sort()
        tags2.sort()

        if tags1 == tags2:
            return True
        else:
            return False
    else:
        return False


def validate(source, translation, validationMode):
    if validationMode != 'HTML':
        if validateXML(source, translation):
            return True
        elif validationMode != 'Auto':
            return False
        else:
            return validateHTML(source, translation)
    else:
        return validateHTML(source, translation)
