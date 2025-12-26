"""
Test script for LLM-based Schema Analyzer and Query Processing

This script tests:
1. Schema analysis with heuristics (no LLM)
2. Query analysis with heuristics (no LLM)
3. Field mapping extraction from sample spreadsheet data
4. End-to-end query processing with dynamic field mappings
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics_service.schema_analyzer import (
    LLMSchemaAnalyzer,
    FieldMapping,
    QueryAnalysis,
    analyze_spreadsheet_schema,
    analyze_user_query
)

# Sample spreadsheet data (mimicking the purchase orders Excel)
SAMPLE_HEADERS = [
    "Purchase Date", "Order ID", "Customer Name", "Category",
    "Product", "Quantity", "Unit Price", "Total Sales",
    "Region", "Sales Rep", "Payment Method", "Status"
]

SAMPLE_DATA = [
    {
        "Purchase Date": "2023-01-15",
        "Order ID": "ORD-001",
        "Customer Name": "Acme Corp",
        "Category": "Electronics",
        "Product": "Laptop",
        "Quantity": 5,
        "Unit Price": 1200.00,
        "Total Sales": 6000.00,
        "Region": "North",
        "Sales Rep": "John Smith",
        "Payment Method": "Credit Card",
        "Status": "Delivered"
    },
    {
        "Purchase Date": "2023-02-20",
        "Order ID": "ORD-002",
        "Customer Name": "TechStart Inc",
        "Category": "Furniture",
        "Product": "Office Chair",
        "Quantity": 20,
        "Unit Price": 250.00,
        "Total Sales": 5000.00,
        "Region": "South",
        "Sales Rep": "Jane Doe",
        "Payment Method": "Bank Transfer",
        "Status": "Shipped"
    },
    {
        "Purchase Date": "2023-03-10",
        "Order ID": "ORD-003",
        "Customer Name": "GreenLeaf LLC",
        "Category": "Stationery",
        "Product": "Printer Paper",
        "Quantity": 100,
        "Unit Price": 25.00,
        "Total Sales": 2500.00,
        "Region": "East",
        "Sales Rep": "Bob Wilson",
        "Payment Method": "Credit Card",
        "Status": "Delivered"
    },
    {
        "Purchase Date": "2024-01-05",
        "Order ID": "ORD-004",
        "Customer Name": "DataFlow Systems",
        "Category": "Electronics",
        "Product": "Monitor",
        "Quantity": 10,
        "Unit Price": 400.00,
        "Total Sales": 4000.00,
        "Region": "West",
        "Sales Rep": "Alice Johnson",
        "Payment Method": "Invoice",
        "Status": "Processing"
    },
    {
        "Purchase Date": "2024-02-28",
        "Order ID": "ORD-005",
        "Customer Name": "SmartTech Co",
        "Category": "Accessories",
        "Product": "Keyboard",
        "Quantity": 50,
        "Unit Price": 80.00,
        "Total Sales": 4000.00,
        "Region": "North",
        "Sales Rep": "John Smith",
        "Payment Method": "Credit Card",
        "Status": "Delivered"
    }
]


def test_schema_analysis_heuristics():
    """Test schema analysis using heuristics (no LLM)."""
    print("\n" + "=" * 70)
    print("TEST 1: Schema Analysis with Heuristics")
    print("=" * 70)

    analyzer = LLMSchemaAnalyzer(llm_client=None)
    mappings = analyzer.analyze_schema(SAMPLE_HEADERS, SAMPLE_DATA, use_llm=False)

    print(f"\nAnalyzed {len(mappings)} columns:")
    print("-" * 70)

    for name, mapping in mappings.items():
        print(f"  {name}:")
        print(f"    semantic_type: {mapping.semantic_type}")
        print(f"    data_type: {mapping.data_type}")
        print(f"    aggregation: {mapping.aggregation}")
        print(f"    description: {mapping.description}")

    # Validate key field mappings
    expected_mappings = {
        "Purchase Date": ("date", "datetime", None),
        "Category": ("category", "string", "group_by"),
        "Total Sales": ("amount", "number", "sum"),
        "Quantity": ("quantity", "number", "sum"),
        "Region": ("region", "string", "group_by"),
        "Sales Rep": ("person", "string", "group_by"),
        "Payment Method": ("method", "string", "group_by"),
        "Status": ("status", "string", "group_by"),
        "Order ID": ("identifier", "string", None),
    }

    print("\n" + "-" * 70)
    print("Validation:")
    all_passed = True
    for field, (exp_type, exp_dtype, exp_agg) in expected_mappings.items():
        if field in mappings:
            m = mappings[field]
            passed = (m.semantic_type == exp_type and
                     m.data_type == exp_dtype and
                     m.aggregation == exp_agg)
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
                print(f"  {status}: {field}")
                print(f"    Expected: type={exp_type}, dtype={exp_dtype}, agg={exp_agg}")
                print(f"    Got:      type={m.semantic_type}, dtype={m.data_type}, agg={m.aggregation}")
            else:
                print(f"  {status}: {field} -> {exp_type}/{exp_dtype}/{exp_agg}")
        else:
            print(f"  ✗ FAIL: {field} not found in mappings")
            all_passed = False

    return all_passed


def test_query_analysis_heuristics():
    """Test query analysis using heuristics (no LLM)."""
    print("\n" + "=" * 70)
    print("TEST 2: Query Analysis with Heuristics")
    print("=" * 70)

    # First get field mappings
    field_mappings = analyze_spreadsheet_schema(SAMPLE_HEADERS, SAMPLE_DATA, llm_client=None)

    # Test queries
    test_queries = [
        ("Summarize total sales by category", "sum", ["Category"], None),
        ("Show sales by year and category", "sum", ["Category"], "yearly"),
        ("Average sales per region", "avg", ["Region"], None),
        ("Monthly revenue breakdown", "sum", [], "monthly"),
        ("Count orders by status", "count", ["Status"], None),
        ("Total quantity by product", "sum", ["Product"], None),
    ]

    all_passed = True

    for query, exp_agg, exp_groups, exp_time in test_queries:
        print(f"\nQuery: \"{query}\"")
        analysis = analyze_user_query(query, field_mappings, llm_client=None)

        print(f"  aggregation_type: {analysis['aggregation_type']}")
        print(f"  group_by_fields: {analysis['group_by_fields']}")
        print(f"  time_grouping: {analysis['time_grouping']}")
        print(f"  explanation: {analysis['explanation']}")

        # Validate
        agg_ok = analysis['aggregation_type'] == exp_agg
        time_ok = analysis['time_grouping'] == exp_time

        # Check groups - at least one expected group should match
        groups_ok = True
        if exp_groups:
            found_groups = [g for g in exp_groups if any(g.lower() in gf.lower() for gf in analysis['group_by_fields'])]
            groups_ok = len(found_groups) > 0 or len(analysis['group_by_fields']) > 0

        passed = agg_ok and time_ok and groups_ok
        if not passed:
            all_passed = False
            print(f"  ✗ VALIDATION FAILED")
            if not agg_ok:
                print(f"    Expected aggregation: {exp_agg}, got: {analysis['aggregation_type']}")
            if not time_ok:
                print(f"    Expected time_grouping: {exp_time}, got: {analysis['time_grouping']}")
            if not groups_ok:
                print(f"    Expected groups: {exp_groups}, got: {analysis['group_by_fields']}")
        else:
            print(f"  ✓ VALIDATION PASSED")

    return all_passed


def test_convenience_functions():
    """Test the convenience wrapper functions."""
    print("\n" + "=" * 70)
    print("TEST 3: Convenience Functions")
    print("=" * 70)

    # Test analyze_spreadsheet_schema
    print("\n1. Testing analyze_spreadsheet_schema()...")
    mappings = analyze_spreadsheet_schema(SAMPLE_HEADERS, SAMPLE_DATA)

    print(f"   Returned {len(mappings)} field mappings")
    print(f"   Keys: {list(mappings.keys())[:5]}...")

    # Verify structure
    for name, mapping in list(mappings.items())[:1]:
        print(f"   Sample mapping for '{name}':")
        print(f"     - semantic_type: {mapping.get('semantic_type')}")
        print(f"     - data_type: {mapping.get('data_type')}")
        print(f"     - aggregation: {mapping.get('aggregation')}")

    # Test analyze_user_query
    print("\n2. Testing analyze_user_query()...")
    query = "Show me total sales grouped by category and year"
    analysis = analyze_user_query(query, mappings)

    print(f"   Query: \"{query}\"")
    print(f"   Analysis result:")
    print(f"     - primary_metric_field: {analysis.get('primary_metric_field')}")
    print(f"     - aggregation_type: {analysis.get('aggregation_type')}")
    print(f"     - group_by_fields: {analysis.get('group_by_fields')}")
    print(f"     - time_grouping: {analysis.get('time_grouping')}")
    print(f"     - explanation: {analysis.get('explanation')}")

    return True


def test_field_mapping_storage_format():
    """Test that field mappings can be stored and retrieved correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: Field Mapping Storage Format")
    print("=" * 70)

    # Generate field mappings
    analyzer = LLMSchemaAnalyzer()
    mappings = analyzer.analyze_schema(SAMPLE_HEADERS, SAMPLE_DATA, use_llm=False)

    # Convert to storage format
    storage_format = analyzer.to_field_mappings_dict(mappings)

    print(f"\nStorage format (JSON-serializable dict):")
    print(f"  Keys: {list(storage_format.keys())}")

    # Verify each mapping has required fields
    required_fields = ['semantic_type', 'data_type', 'aggregation', 'description', 'aliases', 'original_name']
    all_valid = True

    for name, mapping in storage_format.items():
        missing = [f for f in required_fields if f not in mapping]
        if missing:
            print(f"  ✗ {name}: Missing fields: {missing}")
            all_valid = False

    if all_valid:
        print(f"  ✓ All {len(storage_format)} mappings have required fields")

    # Test round-trip: storage format -> query analysis
    print("\n  Testing round-trip (storage -> query analysis)...")
    analysis = analyze_user_query("Total sales by region", storage_format)

    if analysis and 'aggregation_type' in analysis:
        print(f"  ✓ Round-trip successful")
        print(f"    Aggregation: {analysis['aggregation_type']}")
    else:
        print(f"  ✗ Round-trip failed")
        all_valid = False

    return all_valid


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print(" LLM SCHEMA ANALYZER TEST SUITE")
    print("=" * 70)

    results = {}

    results['Schema Analysis (Heuristics)'] = test_schema_analysis_heuristics()
    results['Query Analysis (Heuristics)'] = test_query_analysis_heuristics()
    results['Convenience Functions'] = test_convenience_functions()
    results['Storage Format'] = test_field_mapping_storage_format()

    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("-" * 70)
    print(f"  Total: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
