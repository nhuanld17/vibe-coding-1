#!/usr/bin/env python3
"""
Test tất cả các endpoints của Missing Person AI API.

Endpoints được test:
1. GET  /health                           - Health check
2. GET  /                                 - API info
3. POST /api/v1/upload/missing           - Upload người mất tích
4. POST /api/v1/upload/found             - Upload người tìm thấy
5. GET  /api/v1/search/missing/{case_id} - Tìm kiếm theo case_id
6. GET  /api/v1/search/found/{found_id}  - Tìm kiếm theo found_id
7. GET  /api/v1/search/cases/all         - List tất cả cases
"""

import requests
import json
from pathlib import Path
import time


BASE_URL = "http://localhost:8000"


def print_section(title):
    """In header cho mỗi section."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_response(response, show_full=False):
    """In kết quả response."""
    print(f"\n📊 Status Code: {response.status_code}")
    
    if response.status_code in [200, 201]:
        print("✅ SUCCESS")
        data = response.json()
        if show_full:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            # In tóm tắt
            if 'message' in data:
                print(f"Message: {data['message']}")
            if 'point_id' in data:
                print(f"Point ID: {data['point_id']}")
            if 'case_id' in data:
                print(f"Case ID: {data['case_id']}")
            if 'found_id' in data:
                print(f"Found ID: {data['found_id']}")
            if 'potential_matches' in data:
                matches = data['potential_matches']
                print(f"Potential Matches: {len(matches)}")
                for i, match in enumerate(matches[:3], 1):  # Top 3
                    print(f"  Match {i}: Similarity={match.get('face_similarity', 0):.3f}, "
                          f"Confidence={match.get('confidence_level', 'N/A')}")
    else:
        print("❌ ERROR")
        print(response.text)
    
    return response


def test_1_health_check():
    """Test 1: Health Check."""
    print_section("TEST 1: Health Check")
    
    print(f"GET {BASE_URL}/health")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Services Status:")
        for service, status in data.get('services', {}).items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {service}: {status}")
        
        if 'database_stats' in data and data['database_stats']:
            print(f"\n📊 Database Statistics:")
            stats = data['database_stats']
            if 'missing_persons' in stats:
                print(f"  Missing persons: {stats['missing_persons'].get('points_count', 0)}")
            if 'found_persons' in stats:
                print(f"  Found persons: {stats['found_persons'].get('points_count', 0)}")


def test_2_api_info():
    """Test 2: API Info."""
    print_section("TEST 2: API Info")
    
    print(f"GET {BASE_URL}/")
    response = requests.get(f"{BASE_URL}/", timeout=10)
    print_response(response, show_full=True)


def test_3_upload_missing():
    """Test 3: Upload Missing Person."""
    print_section("TEST 3: Upload Missing Person")
    
    # Tìm ảnh từ dataset
    image_path = Path("datasets/FGNET_organized/person_001/age_02.jpg")
    
    if not image_path.exists():
        print(f"❌ ERROR: Image not found at {image_path}")
        return None
    
    print(f"📷 Image: {image_path}")
    
    # Metadata theo schema mới
    metadata = {
        "case_id": "DEMO_MISSING_001",  # Optional, có thể để None
        "name": "John Doe",
        "age_at_disappearance": 2,
        "year_disappeared": 2020,
        "gender": "male",
        "location_last_seen": "New York, USA",
        "contact": "family@example.com",
        "height_cm": None,  # Optional
        "birthmarks": None,  # Optional list
        "additional_info": "Young child with brown hair"  # Optional
    }
    
    print(f"\n📝 Metadata:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    # Upload
    print(f"\nPOST {BASE_URL}/api/v1/upload/missing")
    
    with open(image_path, 'rb') as img_file:
        files = {'image': (image_path.name, img_file, 'image/jpeg')}
        data = {'metadata': json.dumps(metadata)}
        
        response = requests.post(
            f"{BASE_URL}/api/v1/upload/missing",
            files=files,
            data=data,
            timeout=30
        )
    
    print_response(response)
    
    if response.status_code == 200:
        return response.json()
    return None


def test_4_upload_found(case_id="DEMO_MISSING_001"):
    """Test 4: Upload Found Person (Should match với missing person)."""
    print_section("TEST 4: Upload Found Person")
    
    # Upload ảnh cùng người nhưng lớn tuổi hơn
    image_path = Path("datasets/FGNET_organized/person_001/age_22.jpg")
    
    if not image_path.exists():
        print(f"❌ ERROR: Image not found at {image_path}")
        return None
    
    print(f"📷 Image: {image_path}")
    print(f"(Cùng người với TEST 3 nhưng lớn tuổi hơn - should match!)")
    
    # Metadata theo schema mới
    metadata = {
        "found_id": "DEMO_FOUND_001",  # Optional
        "name": None,  # Optional
        "current_age_estimate": 22,
        "gender": "male",
        "current_location": "Los Angeles, USA",
        "finder_contact": "finder@example.com",
        "visible_marks": None,  # Optional list
        "current_condition": "Good health",  # Optional
        "additional_info": "Adult male found wandering"  # Optional
    }
    
    print(f"\n📝 Metadata:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    # Upload
    print(f"\nPOST {BASE_URL}/api/v1/upload/found")
    
    with open(image_path, 'rb') as img_file:
        files = {'image': (image_path.name, img_file, 'image/jpeg')}
        data = {'metadata': json.dumps(metadata)}
        
        response = requests.post(
            f"{BASE_URL}/api/v1/upload/found",
            files=files,
            data=data,
            timeout=30
        )
    
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        
        # Show detailed matches
        matches = data.get('potential_matches', [])
        if matches:
            print(f"\n🎯 MATCHES FOUND: {len(matches)}")
            for i, match in enumerate(matches, 1):
                print(f"\n--- Match {i} ---")
                print(f"  Case ID: {match.get('metadata', {}).get('case_id', 'N/A')}")
                print(f"  Name: {match.get('metadata', {}).get('name', 'N/A')}")
                print(f"  Face Similarity: {match.get('face_similarity', 0):.4f}")
                print(f"  Confidence Level: {match.get('confidence_level', 'N/A')}")
                print(f"  Confidence Score: {match.get('confidence_score', 0):.4f}")
                print(f"  Contact: {match.get('contact', 'N/A')}")
                if 'explanation' in match:
                    print(f"  Summary: {match['explanation'].get('summary', 'N/A')}")
        
        return data
    return None


def test_5_search_missing(case_id="DEMO_MISSING_001"):
    """Test 5: Search Missing Person by Case ID."""
    print_section("TEST 5: Search Missing Person")
    
    print(f"GET {BASE_URL}/api/v1/search/missing/{case_id}")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/search/missing/{case_id}",
        timeout=10
    )
    
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nRecords found: {data.get('total_found', 0)}")
        
        matches = data.get('matches', [])
        if matches:
            match = matches[0]
            metadata = match.get('metadata', {})
            print(f"\nRecord Details:")
            print(f"  Case ID: {metadata.get('case_id')}")
            print(f"  Name: {metadata.get('name')}")
            print(f"  Age at disappearance: {metadata.get('age_at_disappearance')}")
            print(f"  Location: {metadata.get('location_last_seen')}")
            print(f"  Confidence Score: {match.get('confidence_score', 0):.4f}")
        
        return data
    return None


def test_6_search_found(found_id="DEMO_FOUND_001"):
    """Test 6: Search Found Person by Found ID."""
    print_section("TEST 6: Search Found Person")
    
    print(f"GET {BASE_URL}/api/v1/search/found/{found_id}")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/search/found/{found_id}",
        timeout=10
    )
    
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nRecords found: {data.get('total_found', 0)}")
        
        matches = data.get('matches', [])
        if matches:
            match = matches[0]
            metadata = match.get('metadata', {})
            print(f"\nRecord Details:")
            print(f"  Found ID: {metadata.get('found_id')}")
            print(f"  Current age: {metadata.get('current_age_estimate')}")
            print(f"  Location: {metadata.get('current_location')}")
            print(f"  Confidence Score: {match.get('confidence_score', 0):.4f}")
        
        return data
    return None


def test_7_list_all_cases():
    """Test 7: List All Cases."""
    print_section("TEST 7: List All Cases")
    
    print(f"GET {BASE_URL}/api/v1/search/cases/all")
    
    # Test với default params
    response = requests.get(
        f"{BASE_URL}/api/v1/search/cases/all",
        params={"limit": 50},
        timeout=10
    )
    
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        stats = data.get('statistics', {})
        cases = data.get('cases', {})
        
        print(f"\n📊 Statistics:")
        print(f"  Total Missing: {stats.get('total_missing', 0)}")
        print(f"  Total Found: {stats.get('total_found', 0)}")
        print(f"  Total Cases: {stats.get('total_cases', 0)}")
        print(f"  Processing Time: {data.get('processing_time_ms', 0):.2f} ms")
        
        # Show sample cases
        missing_list = cases.get('missing', [])
        found_list = cases.get('found', [])
        
        if missing_list:
            print(f"\n📋 Sample Missing Cases (showing first 3):")
            for i, case in enumerate(missing_list[:3], 1):
                metadata = case.get('metadata', {})
                print(f"  {i}. {metadata.get('case_id', 'N/A')} - {metadata.get('name', 'N/A')}")
        
        if found_list:
            print(f"\n📋 Sample Found Cases (showing first 3):")
            for i, case in enumerate(found_list[:3], 1):
                metadata = case.get('metadata', {})
                print(f"  {i}. {metadata.get('found_id', 'N/A')} - Age: {metadata.get('current_age_estimate', 'N/A')}")
        
        return data
    return None


def test_8_list_cases_filtered():
    """Test 8: List Cases with Filter."""
    print_section("TEST 8: List Cases with Type Filter")
    
    # Test filter by type
    for case_type in ['missing', 'found']:
        print(f"\nGET {BASE_URL}/api/v1/search/cases/all?type={case_type}")
        
        response = requests.get(
            f"{BASE_URL}/api/v1/search/cases/all",
            params={"limit": 20, "type": case_type},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            print(f"✅ {case_type.upper()} cases: {stats.get('total_cases', 0)}")


def test_9_search_nonexistent():
    """Test 9: Search for Non-existent Record (Should return 404)."""
    print_section("TEST 9: Search Non-existent Record")
    
    fake_id = "NONEXISTENT_ID_12345"
    print(f"GET {BASE_URL}/api/v1/search/missing/{fake_id}")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/search/missing/{fake_id}",
        timeout=10
    )
    
    print_response(response)
    print(f"\n✅ Expected 404 - Got {response.status_code}")


def main():
    """Run all tests."""
    print("=" * 80)
    print(" " * 20 + "MISSING PERSON AI - API TESTS")
    print("=" * 80)
    
    try:
        # Test 1: Health Check
        test_1_health_check()
        time.sleep(1)
        
        # Test 2: API Info
        test_2_api_info()
        time.sleep(1)
        
        # Test 3: Upload Missing Person
        missing_result = test_3_upload_missing()
        time.sleep(2)
        
        # Test 4: Upload Found Person (should match with test 3)
        found_result = test_4_upload_found()
        time.sleep(2)
        
        # Test 5: Search Missing Person
        if missing_result:
            test_5_search_missing("DEMO_MISSING_001")
        time.sleep(1)
        
        # Test 6: Search Found Person
        if found_result:
            test_6_search_found("DEMO_FOUND_001")
        time.sleep(1)
        
        # Test 7: List All Cases
        test_7_list_all_cases()
        time.sleep(1)
        
        # Test 8: List Cases with Filter
        test_8_list_cases_filtered()
        time.sleep(1)
        
        # Test 9: Search Non-existent
        test_9_search_nonexistent()
        
        # Summary
        print_section("SUMMARY")
        print("✅ All tests completed!")
        print("\n📚 Check Swagger UI for more details: http://localhost:8000/docs")
        print("📊 Check Qdrant Dashboard: http://localhost:6333/dashboard")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure Docker is running: docker-compose up -d")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()