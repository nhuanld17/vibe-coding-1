"""Debug test case (0, 2)"""

match_age = 0
query_age_info = 2

match_details = {
    'gender_match': 1.0,
    'age_consistency': 1.0
}

if query_age_info is not None:
    match_details['query_current_age'] = query_age_info

match_metadata = {}
if match_age is not None:
    match_metadata['age_at_disappearance'] = match_age

match = {
    'face_similarity': 0.95,
    'metadata_similarity': 0.8,
    'match_details': match_details,
    'payload': match_metadata
}

print("Match structure:")
print(f"  match_details: {match['match_details']}")
print(f"  payload: {match['payload']}")

# Check logic
match_metadata_check = match.get('payload', {})
match_age_check = match_metadata_check.get('age_at_disappearance')
if match_age_check is None:
    match_age_check = match_metadata_check.get('current_age_estimate')

print(f"\nmatch_age_check: {match_age_check}")

both_children = False
if match_age_check is not None and match_age_check < 18:
    print("  Match age < 18, checking query age...")
    query_age_info_check = match.get('match_details', {}).get('query_current_age')
    if query_age_info_check is None:
        query_age_info_check = match.get('match_details', {}).get('query_age_at_disappearance')
    print(f"  query_age_info_check: {query_age_info_check}")
    if query_age_info_check is not None and query_age_info_check < 18:
        both_children = True
        print("  Both children!")
    elif query_age_info_check is None:
        both_children = True
        print("  Conservative: both children")

print(f"\nResult: both_children = {both_children}")

