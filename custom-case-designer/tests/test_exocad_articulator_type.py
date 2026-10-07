"""Свой тип антагониста в DentalDB: изменённые копии конфигурации exocad (оригиналы не меняются)."""

import importlib.util
import os
import xml.etree.ElementTree as ET

import pytest

_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "exocad_articulator_type.py")
_spec = importlib.util.spec_from_file_location("exocad_articulator_type", _path)
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)

WORK_PARAMS = ('﻿<?xml version="1.0" encoding="utf-8"?>\r\n<Root>\r\n  <TreatmentParameter>\r\n'
               '    <ShortName>AntagonistType</ShortName>\r\n    <Values>\r\n'
               '      <Value disabledOnSingleJaw="true">ArticulatorSendling</Value>\r\n'
               '      <Value disabledOnSingleJaw="true">ArticulatorKlosterneuburg</Value>\r\n'
               '    </Values>\r\n  </TreatmentParameter>\r\n</Root>\r\n')
CUSTOMER = ('<TranslationContainer>\n  <Translations>\n    <Translation>\n      <Keyword>WpfFTP.SignupLink</Keyword>\n'
            '        <Text>http://www.exocad.com</Text>\n    </Translation>  </Translations>\n</TranslationContainer>\n')
MAPPINGS = ('<?xml version="1.0"?>\n<Mappings>\n\t<Antagonists>\n\t\t<Antagonist>\n\t\t\t<Type>ArticulatorSendling</Type>\n'
            '\t\t\t<FolderName>SAM 2P</FolderName>\n\t\t</Antagonist>\n\t</Antagonists>\n\t<Articulators>\n'
            '\t\t<Articulator>\n\t\t\t<FolderName>SAM 2P</FolderName>\n\t\t\t<TypeInXML>SAM_2P</TypeInXML>\n'
            '\t\t\t<Assignments />\n\t\t</Articulator>\n\t</Articulators>\n</Mappings>\n')


@pytest.fixture
def exocad(tmp_path):
    root = tmp_path / "exocad"
    for rel, text in zip(at.FILES, (WORK_PARAMS, CUSTOMER, MAPPINGS)):
        p = root / rel.replace("\\", os.sep)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
    return root


def test_patch_adds_type_label_and_mapping(exocad, tmp_path):
    out = tmp_path / "patch"
    written = at.patch(str(exocad), str(out))
    assert len(written) == 3
    for rel in at.FILES:  # оригиналы не тронуты
        assert (exocad / rel.replace("\\", os.sep)).read_bytes().count(at.TYPE.encode()) == 0
    wp = (out / at.FILES[0].replace("\\", os.sep)).read_bytes()
    assert wp.startswith(b"\xef\xbb\xbf") and b"\r\n" in wp and b"\n" not in wp.replace(b"\r\n", b"")
    values = [v.text for v in ET.fromstring(wp.decode("utf-8-sig")).iter("Value")]
    assert values == ["ArticulatorSendling", "ArticulatorKlosterneuburg", at.TYPE]  # новый — последним в списке
    cust = ET.parse(out / at.FILES[1].replace("\\", os.sep)).getroot()
    labels = {t.findtext("Keyword"): t.findtext("Text") for t in cust.iter("Translation")}
    assert labels[f"AntagonistType.Value.{at.TYPE}"] == at.LABEL
    maps = ET.parse(out / at.FILES[2].replace("\\", os.sep)).getroot()
    assert {(a.findtext("Type"), a.findtext("FolderName")) for a in maps.iter("Antagonist")} >= {(at.TYPE, at.FOLDER)}
    assert at.FOLDER in [a.findtext("FolderName") for a in maps.iter("Articulator")]
    # повторный запуск по уже изменённым файлам ничего не дублирует
    assert at.patch(str(out), str(tmp_path / "again")) == []
