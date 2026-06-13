import re
def coordinate_extractor():
    with open("ProKI_design.xml", "r") as f:
        xml_text = f.read()

    # Find the <Polygon> ... </Polygon> block
    start_tag = "<Profile>"
    end_tag = "</Profile>"

    start_idx = xml_text.find(start_tag)
    end_idx = xml_text.find(end_tag, start_idx)

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Polygon block not found in XML.")

    # Extract snippet including tags
    polygon_snippet = xml_text[start_idx:end_idx + len(end_tag)]

    # Find all x and y values
    coords = re.findall(r'(?:PolyBegin|PolyStepSegment) x="([\d.]+)" y="([\d.]+)"', polygon_snippet)

    x_coords = [float(x) for x, y in coords]
    y_coords = [float(y) for x, y in coords]

    print("x:", x_coords)
    print("y:", y_coords)
    
    return x_coords,y_coords
