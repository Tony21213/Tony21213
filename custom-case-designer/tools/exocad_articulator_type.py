"""Свой тип антагониста в DentalDB — «В артикуляторе Custom Case Designer» (проверено на exocad 3.3 Chemnitz).

Скрипт не меняет exocad: он кладёт изменённые копии трёх файлов конфигурации в отдельную папку (та же структура,
кодировка и переводы строк, что у оригиналов), а подкладывает их пользователь — с резервными копиями:

* DentalDB\\config\\WorkParamsDB.xml — значение ArticulatorCustomCaseDesigner в списке AntagonistType
  (после ArticulatorKlosterneuburg);
* DentalDB\\languages\\customer.xml — подпись пункта (файл подписей клиента, языковые файлы exocad не трогаются);
* DentalCADApp\\config\\articulatormappings.xml — тип → папка library\\articulator\\Custom Case Designer
  (<Antagonists>) и запись артикулятора (<Articulators>).

DentalCAD такой тип принимает: проект с ним открывается, в диалоге «Вирт. артикулятор» сразу выбран наш
артикулятор. Обновление exocad, скорее всего, перезапишет эти файлы — тогда скрипт запускают снова.

Запуск: python tools/exocad_articulator_type.py [папка exocad, где DentalCADApp и DentalDB] [папка для копий]
"""

import os
import sys
import xml.etree.ElementTree as ET

TYPE = "ArticulatorCustomCaseDesigner"
FOLDER = "Custom Case Designer"
LABEL = "В артикуляторе Custom Case Designer"
FILES = (r"DentalDB\config\WorkParamsDB.xml", r"DentalDB\languages\customer.xml",
         r"DentalCADApp\config\articulatormappings.xml")


def _work_params(t: str) -> str:
    nl = "\r\n" if "\r\n" in t else "\n"
    anchor = '<Value disabledOnSingleJaw="true">ArticulatorKlosterneuburg</Value>'
    if t.count(anchor) != 1:
        raise ValueError("в WorkParamsDB.xml нет единственного ArticulatorKlosterneuburg — формат изменился")
    at = t.index(anchor)
    indent = t[t.rfind(nl, 0, at) + len(nl):at]
    end = at + len(anchor)
    return t[:end] + nl + indent + f'<Value disabledOnSingleJaw="true">{TYPE}</Value>' + t[end:]


def _customer(t: str) -> str:
    if t.count("</Translations>") != 1:
        raise ValueError("в customer.xml нет единственного </Translations>")
    add = (f"  <Translation>\n      <Keyword>AntagonistType.Value.{TYPE}</Keyword>\n"
           f"      <Text>{LABEL}</Text>\n    </Translation>\n  ")
    return t.replace("</Translations>", add + "</Translations>")


def _mappings(t: str) -> str:
    if t.count("</Antagonists>") != 1 or t.count("</Articulators>") != 1:
        raise ValueError("в articulatormappings.xml нет разделов <Antagonists> и <Articulators>")
    t = t.replace("</Antagonists>", f"\t<Antagonist>\n\t\t\t<Type>{TYPE}</Type>\n\t\t\t<FolderName>{FOLDER}</FolderName>\n"
                                    "\t\t</Antagonist>\n\t</Antagonists>")
    return t.replace("</Articulators>", f"\t<Articulator>\n\t\t\t<FolderName>{FOLDER}</FolderName>\n"
                                        "\t\t\t<TypeInXML>Custom_Case_Designer</TypeInXML>\n\t\t\t<Assignments />\n"
                                        "\t\t</Articulator>\n\t</Articulators>")


EDITS = dict(zip(FILES, (_work_params, _customer, _mappings)))


def patch(root: str, out: str) -> list[str]:
    """Изменённые копии в out; уже изменённые файлы пропускаются. Возвращает список записанных путей."""
    written = []
    for rel, edit in EDITS.items():
        raw = open(os.path.join(root, rel), "rb").read()
        bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        if TYPE in text:
            print(f"уже есть, пропускаю: {rel}")
            continue
        new = edit(text)
        ET.fromstring(new.encode("utf-8"))  # копия должна оставаться правильным XML
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write((b"\xef\xbb\xbf" if bom else b"") + new.encode("utf-8"))
        written.append(dst)
        print(f"записано: {dst}")
    return written


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else r"C:\Exo\3.3\exocad-DentalCAD3.3-SR1-2026-01-16"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "exocad-config-patch")
    patch(root, out)
