import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
import json
import os
import re

# ================= Configuration & Setup =================

# 1. Initialize Embedding Model
try:
    from sentence_transformers import SentenceTransformer
    # Using a multilingual model suitable for historical texts (Italian/English/Venetian)
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    HAS_MODEL = True
    print("[Init] Model loaded successfully.")
except ImportError:
    HAS_MODEL = False
    print("[Init] Warning: sentence-transformers not found. Embeddings will be empty vectors.")

# 2. Constants
# Chunk size limit (characters). Allow space for the "Semantic Anchor" header.
CHUNK_SIZE_LIMIT = 600 

# Mapping 1740 Sestiere abbreviations to full names
SESTIERE_MAP = {
    "SM": "San Marco", 
    "CS": "Castello", 
    "CC": "Cannaregio", 
    "CN": "Cannaregio",
    "SP": "San Polo", 
    "SC": "Santa Croce", 
    "DD": "Dorsoduro", 
    "GH": "Ghetto"
}

# ================= Helper Functions =================

def get_embedding(text):
    """
    Generate vector embedding for the input text using the loaded model.
    Returns a zero-filled list if model is not loaded.
    """
    if HAS_MODEL and text and isinstance(text, str):
        return model.encode(text).tolist()
    # Return placeholder vector (dimension 384 depends on MiniLM model)
    return [0.0] * 384

def clean_val(val):
    """
    Utility to clean raw values.
    Handles lists (joins them), None/NaN values, and strips whitespace.
    """
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return None
        clean_list = [str(v) for v in val if v is not None and str(v).lower() not in ['nan', 'null', 'none', '[]']]
        return ", ".join(clean_list) if clean_list else None
    
    if pd.isna(val) or val == "" or str(val).lower() in ['nan', 'null', 'none', '[]']:
        return None
    return str(val).strip()

def split_text(text, limit=CHUNK_SIZE_LIMIT):
    """
    Splits text into chunks respecting sentence boundaries (. ; ! ?).
    Prevents cutting sentences in the middle for better semantic understanding.
    """
    if not text or len(text) <= limit:
        return [text] if text else []
    
    # Split by punctuation followed by space
    sentences = re.split(r'(?<=[.;!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        if current_len + len(sent) > limit:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = len(sent)
        else:
            current_chunk.append(sent)
            current_len += len(sent)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# ================= PHASE 1: Process 1808 (Hybrid & Chunking) =================

def generate_semantic_anchor_1808(texts_list, people_list):
    """
    Generates a 'Semantic Anchor' (Header).
    Example: "This is a residential house owned by the Balbi family located in San Marco."
    """
    # Determine dominant function
    qualities = set()
    for t in texts_list:
        q = clean_val(t.get('quality'))
        if q: qualities.add(q)
    quality_str = ", ".join(list(qualities)[:2]) if qualities else "property"
    
    # Determine owners (Families)
    families = set()
    for p in people_list:
        fam = clean_val(p.get('own_family'))
        if fam: families.add(fam)
    owner_str = f" owned by the {', '.join(families)} family" if families else ""
    
    # Determine location
    district = "Venice"
    if texts_list:
        d = clean_val(texts_list[0].get('district'))
        if d: district = d
        
    return f"This is a {quality_str}{owner_str} located in {district}."

def generate_detailed_body_1808(t_item):
    """
    Converts a single raw text entry into a fluent descriptive paragraph.
    Retains ALL attributes found in the source file for full context.
    """
    parts = []
    
    # Location
    place = clean_val(t_item.get('place'))
    if place: parts.append(f"Located at {place}.")
    
    # IDs (Useful for search validation)
    p_num = clean_val(t_item.get('parcel_number'))
    h_num = clean_val(t_item.get('house_number'))
    struct_desc = []
    if p_num: struct_desc.append(f"Parcel ID {p_num}")
    if h_num: struct_desc.append(f"House No. {h_num}")
    if struct_desc: parts.append(", ".join(struct_desc) + ".")

    # Usage & Rights
    own_types = clean_val(t_item.get('ownership_types'))
    owner_right = clean_val(t_item.get('owner_right_of_use'))
    usage_desc = []
    if own_types: usage_desc.append(f"ownership type: {own_types}")
    if owner_right: usage_desc.append(f"right of use: {owner_right}")
    if usage_desc: parts.append("Legal status: " + "; ".join(usage_desc) + ".")

    # Historical Owners (Transcription) & Previous Entity
    owner_raw = clean_val(t_item.get('owner_transcription'))
    if owner_raw: parts.append(f"Original owner record: {owner_raw}.")
    
    old_ent = clean_val(t_item.get('old_entity'))
    if old_ent: parts.append(f"Previously owned by: {old_ent}.")
    
    # Notes
    supp = clean_val(t_item.get('owner_supplementary'))
    if supp: parts.append(f"Notes: {supp}.")
    
    return " ".join(parts)

def process_1808_integrated(json_path):
    print(f"\n[Phase 1] Processing 1808 Data: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    processed_rows = []
    napo_lookup = {} # Dictionary to store 1808 info for linking with 1740
    
    print(f"Total entries to process: {len(data)}")
    
    for idx, entry in enumerate(data):
        texts_list = entry.get('text', [])
        people_list = entry.get('people', [])
        geoms_data = entry.get('geometries', {})
        features = geoms_data.get('features', [])
        
        # --- A. Generate Semantic Anchor ---
        anchor_text = generate_semantic_anchor_1808(texts_list, people_list)
        
        # --- B. Geometry Aggregation (The Logic Change) ---
        geo_wkt = None
        geo_summary_text = ""
        geo_props_list = [] # Store properties of all sub-parts
        
        if features:
            polygons = []
            types_list = []
            
            for feat in features:
                # 1. Collect Shapes
                if feat.get('geometry'):
                    s = shape(feat['geometry'])
                    polygons.append(s)
                
                # 2. Collect Metadata (Type, Area, etc.)
                props = feat.get('properties', {})
                geo_props_list.append(props)
                
                g_type = clean_val(props.get('geometry_type'))
                if g_type: types_list.append(g_type)
            
            # 3. Create MultiPolygon
            if polygons:
                # unary_union handles overlapping or disjoint polygons gracefully
                combined_geo = unary_union(polygons)
                # Ensure it's a MultiPolygon for consistency in DB
                if combined_geo.geom_type == 'Polygon':
                    combined_geo = MultiPolygon([combined_geo])
                geo_wkt = combined_geo.wkt
            
            # 4. Generate Natural Language Summary of Geometry
            # Example: "comprises 2 structures: a building and a courtyard" "located in the parish of ..."
            first_parish = clean_val(features[0].get('properties', {}).get('parish_standardised'))
            
            structure_str = ""
            if types_list:
                from collections import Counter
                type_counts = Counter(types_list)
                desc_parts = [f"{count} {t}" for t, count in type_counts.items()]
                structure_str = f"The property comprises {len(polygons)} parts: {', '.join(desc_parts)}."

            parish_str = ""
            if first_parish:
                parish_str = f"It is located in the parish of {first_parish}."
            
            # Combine parts
            summary_parts = [s for s in [structure_str, parish_str] if s]
            if summary_parts:
                geo_summary_text = " ".join(summary_parts)
            else:
                geo_summary_text = "Map geometry available."
        
        else:
            # No geometry case
            geo_summary_text = "Map geometry not available."
            features = [{'id': f"no_geo_{idx}"}] # Dummy for ID generation

        # --- C. Generate Text Chunks & Lookup Info ---
        text_chunks = []
        
        if not texts_list:
            text_chunks.append("Map record only. No textual details available.")
        else:
            for t_item in texts_list:
                # 1. Build Body
                body_text = generate_detailed_body_1808(t_item)
                # 2. Split Body
                sub_chunks = split_text(body_text, limit=CHUNK_SIZE_LIMIT)
                text_chunks.extend(sub_chunks)
                
                # 3. Populate Napo Lookup (Parcel Number -> Anchor + Brief Desc)
                p_num = clean_val(t_item.get('parcel_number'))
                if p_num:
                    # We store a summary for 1740 to use
                    napo_lookup[str(p_num)] = f"{anchor_text} {body_text[:150]}..."

        # --- D. Prepare Structured Metadata for SQL Filtering ---
        # Extract unique districts, families, and parcel IDs for hard filtering
        meta_districts = list(set([clean_val(t.get('district')) for t in texts_list if clean_val(t.get('district'))]))
        meta_families = list(set([clean_val(p.get('own_family')) for p in people_list if clean_val(p.get('own_family'))]))
        meta_parcel_nums = list(set([clean_val(t.get('parcel_number')) for t in texts_list if clean_val(t.get('parcel_number'))]))

        # Dump full original data for frontend display
        row_metadata = {
            'year': 1808,
            'filter_districts': meta_districts,
            'filter_families': meta_families,
            'filter_parcel_ids': meta_parcel_nums,
            'geo_props': geo_props_list,
            'raw_text': texts_list,
            'raw_people': people_list
        }

        # Use the ID of the first feature as the main ID, or a fallback
        main_id = str(features[0].get('id', f"agg_{idx}"))

        # --- E. Row Generation ---
        # If text is split into chunks, we create multiple rows sharing the SAME MultiPolygon.
        # This is necessary for vector search (chunking), but they all point to the same "Entry".
        
        for chunk_i, chunk_body in enumerate(text_chunks):
            
            # COMBINE: Anchor + Geo Summary + Chunk Details
            # Ex: "This is a house... comprises a building and courtyard. Details: Located at..."
            final_text = f"{anchor_text} {geo_summary_text} Details: {chunk_body}"
            
            processed_rows.append({
                'source_dataset': '1808_sommarioni',
                'original_id': main_id,
                'chunk_id': f"{main_id}_{chunk_i}",
                'text_representation': final_text,
                'embedding': get_embedding(final_text),
                'geometry': geo_wkt, # The combined MultiPolygon
                'metadata': json.dumps(row_metadata)
            })

        if idx % 500 == 0:
            print(f"Processed {idx} entries...", end='\r')

    # Save to Parquet
    df = pd.DataFrame(processed_rows)
    output_file = 'processed_1808.parquet'
    df.to_parquet(output_file)
    print(f"\n[Phase 1] Done. Saved {len(df)} rows to {output_file}. Lookup size: {len(napo_lookup)}")
    return napo_lookup

# ================= PHASE 2: Process 1740 (Linked & Geo) =================
def generate_semantic_anchor_1740(row):
    """
    Generates a Semantic Anchor (Header) for 1740 records.
    Ensures that every chunk knows 'What', 'Where', and 'Who'.
    """
    # 1. Location
    sestiere = SESTIERE_MAP.get(clean_val(row.get('sestiere')), row.get('sestiere'))
    parish = clean_val(row.get('parish_std'))
    
    loc_str = "Venice"
    if sestiere: loc_str = sestiere
    if parish: loc_str += f" ({parish})"

    # 2. Function (What)
    # Prefer standardized TOP function, fallback to raw
    func = clean_val(row.get('PP_Function_TOP'))
    if not func: func = clean_val(row.get('function'))
    if not func: func = "Property"
    
    # 3. Owner (Who)
    # Prefer standardized Name, fallback to raw
    owner_first = clean_val(row.get('PP_Owner_FirstName'))
    owner_last = clean_val(row.get('PP_Owner_LastName'))
    if owner_last:
        owner = f"{owner_first} {owner_last}".strip()
    else:
        owner = clean_val(row.get('owner_name'))
    
    owner_str = f" owned by {owner}" if owner else ""

    return f"This is a {func}{owner_str} located in {loc_str}."


def process_1740_integrated(tsv_path, geojson_path, napo_lookup):
    print(f"\n[Phase 2] Processing 1740 Catastici Data: {tsv_path}")
    
    # 1. Load Data
    df_text = pd.read_csv(tsv_path, sep='\t')
    df_text['uid'] = df_text['uid'].astype(str) # Ensure string for join
    
    # 2. Load and Prepare Geometry
    if os.path.exists(geojson_path):
        print(f"Loading GeoJSON: {geojson_path}")
        gdf_geo = gpd.read_file(geojson_path)
        
        # Fix CRS: The file is EPSG:32633, we need EPSG:4326 for OpenStreetMap
        if gdf_geo.crs and gdf_geo.crs.to_string() != "EPSG:4326":
            print(f"Converting CRS from {gdf_geo.crs} to EPSG:4326...")
            gdf_geo = gdf_geo.to_crs("EPSG:4326")
        elif gdf_geo.crs is None:
             # If strictly 32633 but missing metadata, force set then convert
             # But here we assume if missing, we default or skip. 
             pass
        
        if 'uid' in gdf_geo.columns:
            gdf_geo['uid'] = gdf_geo['uid'].astype(str)
            # Merge: Right join on text means we keep all text records. 
            merged_df = gdf_geo.merge(df_text, on='uid', how='right', suffixes=('_geo', '')) 
        else:
            print("Error: 'uid' not found in GeoJSON. Proceeding with text only.")
            merged_df = df_text
            merged_df['geometry'] = None
    else:
        print("GeoJSON not found. Proceeding with text only.")
        merged_df = df_text
        merged_df['geometry'] = None

    processed_rows = []
    
    print(f"Generating 1740 entries with Linkage...")
    for idx, row in merged_df.iterrows():
        # --- A. Generate Anchor (Header) ---
        anchor_text = generate_semantic_anchor_1740(row)

        # --- B. Extract Attributes ---
        sestiere = SESTIERE_MAP.get(clean_val(row.get('sestiere')), row.get('sestiere'))
        place = clean_val(row.get('place'))
        parish = clean_val(row.get('parish_std'))
        
        # Function & Description
        func = clean_val(row.get('function'))
        func_top = clean_val(row.get('PP_Function_TOP'))
        func_mid = clean_val(row.get('PP_Function_MID'))
        func_prop = clean_val(row.get('PP_Function_PROPERTY'))
        func_geo = clean_val(row.get('PP_Function_GEOMETRY'))

        # Bottega
        bot_trad = clean_val(row.get('PP_Bottega_TRAD'))
        bot_meta = clean_val(row.get('PP_Bottega_METACATEGORY'))

        # Economics
        rent = clean_val(row.get('an_rendi'))
        quan_income = clean_val(row.get('quantity_income'))
        qual_income = clean_val(row.get('quality_income'))

        # People
        tenant = clean_val(row.get('ten_name'))
        owner_orig = clean_val(row.get('owner_name'))
        owner_std_first = clean_val(row.get('PP_Owner_FirstName'))
        owner_std_last = clean_val(row.get('PP_Owner_LastName'))
        owner_std = f"{owner_std_first} {owner_std_last}".strip() if owner_std_last else None

        own_title = clean_val(row.get('PP_Owner_Title'))
        own_code_simp = clean_val(row.get('PP_OwnerCode_SIMPL'))
        own_prof = clean_val(row.get('owner_mestiere_std'))
        own_entity = clean_val(row.get('PP_Owner_Entity'))
        own_notes = clean_val(row.get('PP_Owner_Notes'))
        
        # Linking ID
        id_napo = clean_val(row.get('id_napo'))

        uid = str(row.get('uid', idx))

        # --- C. Construct Text ---
        sentences = []
        
        # Context
        loc_parts = []
        if sestiere: loc_parts.append(f"District: {sestiere}")
        if parish: loc_parts.append(f"Parish: {parish}")
        if place: loc_parts.append(f"Place: {place}")
        if loc_parts: sentences.append(". ".join(loc_parts) + ".")
        
        # Function
        func_parts = []
        if func: func_parts.append(f"Function: {func}")
        if func_top or func_mid:
            h_str = " > ".join([x for x in [func_top, func_mid] if x])
            func_parts.append(f"Class: {h_str}")
        feats = [x for x in [func_prop, func_geo] if x]
        if feats: func_parts.append(f"Features: {', '.join(feats)}")
        if bot_trad or bot_meta:
            shop_str = f"Trade: {bot_trad}" if bot_trad else "Shop"
            if bot_meta: shop_str += f" ({bot_meta})"
            func_parts.append(shop_str)
        if func_parts: sentences.append("; ".join(func_parts) + ".")
        
        # Economics
        econ_parts = []
        if rent: econ_parts.append(f"Annual Rent: {rent}")
        if quan_income or qual_income:
            inc = f"{quan_income or ''} {qual_income or ''}".strip()
            econ_parts.append(f"Income: {inc}")
        if econ_parts: sentences.append(". ".join(econ_parts) + ".")
        
        # People
        ppl_parts = []
        if tenant: ppl_parts.append(f"Tenant: {tenant}")
        
        owner_desc = []
        if owner_orig: owner_desc.append(f"Original Name: {owner_orig}")
        if owner_std: owner_desc.append(f"Standardised Name: {owner_std}")
        if own_title: owner_desc.append(f"Title: {own_title}")
        if owner_desc: ppl_parts.append(f"Owner: {' | '.join(owner_desc)}")
        
        own_dets = []
        if own_prof: own_dets.append(f"Profession: {own_prof}")
        if own_code_simp: own_dets.append(f"Type: {own_code_simp}")
        if own_entity: own_dets.append(f"Entity: {own_entity}")
        if own_notes: own_dets.append(f"Notes: {own_notes}")
        if own_dets: ppl_parts.append(f"Owner Details: {', '.join(own_dets)}")
        
        if ppl_parts: sentences.append(". ".join(ppl_parts) + ".")
        
        # --- D. LINKAGE INJECTION (The Hybrid Magic) ---
        if id_napo and str(id_napo) in napo_lookup:
            # Inject 1808 knowledge into 1740 record
            linked_info = napo_lookup[str(id_napo)]
            sentences.append(f"Future Reference (1808 Link): Linked to Parcel {id_napo}. {linked_info}")
        elif id_napo:
            sentences.append(f"Future Reference (1808 Link): Linked to Parcel {id_napo}.")

        full_body_text = " ".join(sentences)

        # --- E. Save Data ---
        chunks = split_text(full_body_text, limit=CHUNK_SIZE_LIMIT)

        geo_wkt = None
        if 'geometry' in row and row.geometry is not None:
             geo_wkt = row.geometry.wkt
        
        # Clean metadata (remove geometry objects to allow JSON serialization)
        meta_dict = {k: v for k, v in row.items() if k != 'geometry' and clean_val(v) is not None}
        
        for i, chunk in enumerate(chunks):
            # For 1740, the "chunk" is often just the whole text, but if split,
            # we ensure the first part of context (e.g. "Record from 1740...") implies continuity.
            # split_text_smartly handles simple splitting.
            final_text = f"{anchor_text} Details: {chunk}"

            processed_rows.append({
                'source_dataset': '1740_catastici',
                'original_id': uid,
                'chunk_id': f"{uid}_{i}",
                'text_representation': final_text, # The chunked text
                'embedding': get_embedding(final_text),
                'geometry': geo_wkt, # Duplicated geometry for each chunk
                'metadata': json.dumps(meta_dict)
            })
        
        if idx % 500 == 0:
            print(f"Processed {idx} rows...", end='\r')

    # Save to Parquet
    df = pd.DataFrame(processed_rows)
    output_file = 'processed_1740.parquet'
    df.to_parquet(output_file)
    print(f"\n[Phase 2] Done. Saved {len(df)} rows to {output_file}.")

# ================= MAIN EXECUTION BLOCK =================

if __name__ == "__main__":
    # Define file paths
    FILE_1808_AGG = "venice-1808-landregister/venice_1808_landregister_aggregated_data.json"
    FILE_1740_TSV = "venice-1740-landregister/1740_Catastici_2025-09-24.tsv"
    FILE_1740_GEO = "venice-1740-landregister/1740_Catastici_2025-09-24.geojson"
    PARQUET_1808 = "processed_1808.parquet"

    # Step 1: Process 1808 first to build the Linkage Lookup Table
    napo_lookup_map = {}
    if os.path.exists(FILE_1808_AGG):
        napo_lookup_map = process_1808_integrated(FILE_1808_AGG)
    else:
        print(f"Critical Error: 1808 File not found: {FILE_1808_AGG}")

    # Step 2: Process 1740 using the Lookup Table
    if os.path.exists(FILE_1740_TSV):
        process_1740_integrated(FILE_1740_TSV, FILE_1740_GEO, napo_lookup_map)
    else:
        print(f"Critical Error: 1740 TSV not found: {FILE_1740_TSV}")

    print("\nAll processing complete. Ready for Database Ingestion.")