"""Build data/countries.geojson: the 193 UN member states + Vatican + Palestine, from Natural Earth 1:10m.

Territorial choices: Northern Cyprus is unioned into Cyprus and Somaliland into Somalia (UN-recognised
territory of those states); Western Sahara, Kosovo and Taiwan are not UN members and are left out.
"""

import json
import sys
import urllib.request
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain_worldle.game.countries import COUNTRIES_PATH, main_territory

SOURCE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
SIMPLIFY_DEG = 0.01
MERGE_INTO = {"N. Cyprus": "CYP", "Somaliland": "SOM"}

UN_MEMBERS = """
AFG ALB DZA AND AGO ATG ARG ARM AUS AUT AZE BHS BHR BGD BRB BLR BEL BLZ BEN BTN BOL BIH BWA BRA BRN BGR BFA BDI
CPV KHM CMR CAN CAF TCD CHL CHN COL COM COG COD CRI CIV HRV CUB CYP CZE DNK DJI DMA DOM ECU EGY SLV GNQ ERI EST
SWZ ETH FJI FIN FRA GAB GMB GEO DEU GHA GRC GRD GTM GIN GNB GUY HTI HND HUN ISL IND IDN IRN IRQ IRL ISR ITA JAM
JPN JOR KAZ KEN KIR PRK KOR KWT KGZ LAO LVA LBN LSO LBR LBY LIE LTU LUX MDG MWI MYS MDV MLI MLT MHL MRT MUS MEX
FSM MDA MCO MNG MNE MAR MOZ MMR NAM NRU NPL NLD NZL NIC NER NGA MKD NOR OMN PAK PLW PAN PNG PRY PER PHL POL PRT
QAT ROU RUS RWA KNA LCA VCT WSM SMR STP SAU SEN SRB SYC SLE SGP SVK SVN SLB SOM ZAF SSD ESP LKA SDN SUR SWE CHE
SYR TJK TZA THA TLS TGO TON TTO TUN TUR TKM TUV UGA UKR ARE GBR USA URY UZB VUT VEN VNM YEM ZMB ZWE
VAT PSE
""".split()


def code_of(props: dict) -> str:
    return props["ISO_A3"] if props["ISO_A3"] != "-99" else props["ADM0_A3"]


def main() -> None:
    assert len(UN_MEMBERS) == 195 and len(set(UN_MEMBERS)) == 195
    with urllib.request.urlopen(SOURCE) as r:
        source = json.load(r)

    features = {}
    extras = {}
    for ft in source["features"]:
        props, code = ft["properties"], code_of(ft["properties"])
        if code in UN_MEMBERS:
            features[code] = (props, shape(ft["geometry"]))
        elif props["NAME"] in MERGE_INTO:
            extras.setdefault(MERGE_INTO[props["NAME"]], []).append(shape(ft["geometry"]))
        else:
            print(f"dropped: {props['NAME']} ({code})")

    missing = [c for c in UN_MEMBERS if c not in features]
    assert not missing, f"missing from Natural Earth: {missing}"

    out = []
    for code in UN_MEMBERS:
        props, geom = features[code]
        if code in extras:
            geom = unary_union([geom, *extras[code]])
            print(f"merged into {props['NAME']}: {[n for n, c in MERGE_INTO.items() if c == code]}")
        geom = main_territory(geom).simplify(SIMPLIFY_DEG, preserve_topology=True)
        out.append(
            {
                "type": "Feature",
                "properties": {"NAME": props["NAME"], "ISO_A3": code, "CONTINENT": props["CONTINENT"], "LABEL_X": props["LABEL_X"], "LABEL_Y": props["LABEL_Y"]},
                "geometry": mapping(geom),
            }
        )

    with open(COUNTRIES_PATH, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": out}, f, separators=(",", ":"))
    print(f"wrote {len(out)} countries to {COUNTRIES_PATH} ({COUNTRIES_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
