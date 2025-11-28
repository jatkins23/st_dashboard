#!/usr/bin/env python3
"""Test script for refactored query and result classes.

Tests the new architecture:
- Query classes in streettransformer.query
- Result classes in dashboard.backend.results
- Dashboard search function wrappers
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from streettransformer import (
    Config,
    EmbeddingDB,
    StateLocationQuery,
    ChangeLocationQuery,
    StateTextQuery,
    CLIPEncoder
)
from dashboard.backend import (
    search_by_location,
    search_by_text,
    search_change_patterns,
    StateLocationResult,
    ChangeLocationResult
)
from streettransformer.database import get_connection


class TestRefactor:
    """Test suite for refactored architecture."""

    def __init__(self):
        """Initialize test with existing database."""
        self.config = Config(
            database_path="/Users/jon/code/st_preprocessing/core.ddb",
            universe_name="lion"
        )
        self.db = EmbeddingDB(self.config)
        self.db_connection_func = lambda: get_connection(
            self.config.database_path,
            read_only=True
        )

        # Get test data
        self.years = self.db.get_years()
        if not self.years:
            raise RuntimeError("No years found in database")
        self.test_year = self.years[0]

        # Get a test location_id
        with self.db_connection_func() as con:
            sample = con.execute(f"""
                SELECT location_id
                FROM lion.image_embeddings
                WHERE year = {self.test_year}
                    AND embedding IS NOT NULL
                LIMIT 1
            """).df()

        if sample.empty:
            raise RuntimeError(f"No embeddings found for year {self.test_year}")
        self.test_location_id = int(sample.iloc[0]['location_id'])

        print(f"Test setup complete:")
        print(f"  Database: {self.config.database_path}")
        print(f"  Universe: {self.config.universe_name}")
        print(f"  Test year: {self.test_year}")
        print(f"  Test location: {self.test_location_id}")
        print()

    def test_state_location_query_class(self):
        """Test StateLocationQuery class instantiation and execution."""
        print("Test 1: StateLocationQuery class")
        print("-" * 50)

        try:
            # Create query
            query = StateLocationQuery(
                location_id=self.test_location_id,
                year=self.test_year,
                config=self.config,
                db=self.db,
                limit=5,
                use_faiss=False,  # Use DB search for simplicity
                use_whitening=False,
                db_connection_func=self.db_connection_func
            )
            print("✓ Query instantiated")

            # Execute query
            results = query.execute()
            print(f"✓ Query executed: {len(results)} results")

            # Verify results
            assert isinstance(results, pd.DataFrame), "Results should be DataFrame"
            assert not results.empty, "Results should not be empty"
            assert 'location_id' in results.columns, "Results should have location_id"
            assert 'similarity' in results.columns, "Results should have similarity"
            print(f"✓ Results validated")

            # Check cache key
            cache_key = query.get_cache_key()
            print(f"✓ Cache key: {cache_key}")

            print("✅ PASSED\n")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            return False

    def test_change_location_query_class(self):
        """Test ChangeLocationQuery class instantiation and execution."""
        print("Test 2: ChangeLocationQuery class")
        print("-" * 50)

        try:
            # Need at least 2 years
            if len(self.years) < 2:
                print("⚠ SKIPPED: Need at least 2 years for change detection\n")
                return True

            year_from = self.years[0]
            year_to = self.years[1]

            # Create query
            query = ChangeLocationQuery(
                location_id=self.test_location_id,
                start_year=year_from,
                end_year=year_to,
                config=self.config,
                db=self.db,
                limit=5,
                db_connection_func=self.db_connection_func
            )
            print("✓ Query instantiated")

            # Execute query
            results = query.execute()
            print(f"✓ Query executed: {len(results)} results")

            # Verify results (may be empty if no change vectors)
            assert isinstance(results, pd.DataFrame), "Results should be DataFrame"
            if not results.empty:
                assert 'location_id' in results.columns, "Results should have location_id"
                print("✓ Results validated")
            else:
                print("⚠ No change results (this may be expected)")

            # Check cache key
            cache_key = query.get_cache_key()
            print(f"✓ Cache key: {cache_key}")

            print("✅ PASSED\n")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            return False

    def test_state_text_query_class(self):
        """Test StateTextQuery class instantiation and execution."""
        print("Test 3: StateTextQuery class")
        print("-" * 50)

        try:
            # Create CLIP encoder
            try:
                encoder = CLIPEncoder()
                print("✓ CLIPEncoder created")
            except ImportError:
                print("⚠ SKIPPED: open-clip-torch not installed\n")
                return True

            # Create query
            query = StateTextQuery(
                text_query="street with trees",
                config=self.config,
                db=self.db,
                clip_encoder=encoder,
                year=self.test_year,
                limit=5,
                use_faiss=False,
                use_whitening=False,
                db_connection_func=self.db_connection_func
            )
            print("✓ Query instantiated")

            # Execute query
            results = query.execute()
            print(f"✓ Query executed: {len(results)} results")

            # Verify results
            assert isinstance(results, pd.DataFrame), "Results should be DataFrame"
            assert not results.empty, "Results should not be empty"
            assert 'location_id' in results.columns, "Results should have location_id"
            print("✓ Results validated")

            # Check cache key
            cache_key = query.get_cache_key()
            print(f"✓ Cache key: {cache_key}")

            print("✅ PASSED\n")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            return False

    def test_dashboard_search_by_location(self):
        """Test dashboard search_by_location wrapper."""
        print("Test 4: Dashboard search_by_location wrapper")
        print("-" * 50)

        try:
            # Call dashboard wrapper
            results = search_by_location(
                config=self.config,
                db=self.db,
                db_connection_func=self.db_connection_func,
                location_id=self.test_location_id,
                year=self.test_year,
                limit=5,
                use_faiss=False,
                use_whitening=False
            )
            print(f"✓ Search executed: {len(results)} results")

            # Verify enrichment (should have street names if available)
            assert isinstance(results, pd.DataFrame), "Results should be DataFrame"
            assert not results.empty, "Results should not be empty"
            print("✓ Results validated")

            print("✅ PASSED\n")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            return False

    def test_dashboard_search_change_patterns(self):
        """Test dashboard search_change_patterns wrapper."""
        print("Test 5: Dashboard search_change_patterns wrapper")
        print("-" * 50)

        try:
            # Need at least 2 years
            if len(self.years) < 2:
                print("⚠ SKIPPED: Need at least 2 years for change detection\n")
                return True

            year_from = self.years[0]
            year_to = self.years[1]

            # Call dashboard wrapper
            results = search_change_patterns(
                config=self.config,
                db=self.db,
                db_connection_func=self.db_connection_func,
                location_id=self.test_location_id,
                year_from=year_from,
                year_to=year_to,
                limit=5
            )
            print(f"✓ Search executed: {len(results)} results")

            # Verify results (may be empty)
            assert isinstance(results, pd.DataFrame), "Results should be DataFrame"
            print("✓ Results validated")

            print("✅ PASSED\n")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            return False

    def test_dashboard_search_by_text(self):
        """Test dashboard search_by_text wrapper."""
        print("Test 6: Dashboard search_by_text wrapper")
        print("-" * 50)

        try:
            # Call dashboard wrapper (will create encoder internally)
            try:
                results = search_by_text(
                    config=self.config,
                    db=self.db,
                    db_connection_func=self.db_connection_func,
                    text_query="urban street",
                    year=self.test_year,
                    limit=5,
                    use_faiss=False,
                    use_whitening=False
                )
                print(f"✓ Search executed: {len(results)} results")

                # Verify results
                assert isinstance(results, pd.DataFrame), "Results should be DataFrame"
                assert not results.empty, "Results should not be empty"
                print("✓ Results validated")

                print("✅ PASSED\n")
                return True

            except ImportError:
                print("⚠ SKIPPED: open-clip-torch not installed\n")
                return True

        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            return False

    def test_state_location_result_class(self):
        """Test StateLocationResult class instantiation."""
        print("Test 7: StateLocationResult class")
        print("-" * 50)

        try:
            # Create mock results
            mock_results = pd.DataFrame({
                'location_id': [1, 2, 3],
                'year': [self.test_year] * 3,
                'similarity': [0.95, 0.92, 0.88],
                'location_key': ['key1', 'key2', 'key3']
            })

            # Create result object
            result = StateLocationResult(
                results=mock_results,
                location_id=self.test_location_id,
                year=self.test_year,
                config=self.config,
                db=self.db,
                db_connection_func=self.db_connection_func
            )
            print("✓ Result object instantiated")

            # Test methods
            assert hasattr(result, 'render_accordion'), "Should have render_accordion method"
            assert hasattr(result, 'render_table'), "Should have render_table method"
            assert hasattr(result, 'render_json'), "Should have render_json method"
            print("✓ Methods available")

            # Test JSON rendering
            json_data = result.render_json()
            assert isinstance(json_data, list), "render_json should return list"
            print("✓ JSON rendering works")

            print("✅ PASSED\n")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            return False

    def test_change_location_result_class(self):
        """Test ChangeLocationResult class instantiation."""
        print("Test 8: ChangeLocationResult class")
        print("-" * 50)

        try:
            if len(self.years) < 2:
                print("⚠ SKIPPED: Need at least 2 years\n")
                return True

            year_from = self.years[0]
            year_to = self.years[1]

            # Create mock results
            mock_results = pd.DataFrame({
                'location_id': [1, 2, 3],
                'year_from': [year_from] * 3,
                'year_to': [year_to] * 3,
                'similarity': [0.95, 0.92, 0.88],
                'location_key': ['key1', 'key2', 'key3']
            })

            # Create result object
            result = ChangeLocationResult(
                results=mock_results,
                location_id=self.test_location_id,
                start_year=year_from,
                end_year=year_to,
                config=self.config,
                db=self.db,
                db_connection_func=self.db_connection_func
            )
            print("✓ Result object instantiated")

            # Test methods
            assert hasattr(result, 'render_accordion'), "Should have render_accordion method"
            assert hasattr(result, 'render_table'), "Should have render_table method"
            print("✓ Methods available")

            print("✅ PASSED\n")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            return False

    def run_all_tests(self):
        """Run all tests and report results."""
        print("=" * 50)
        print("REFACTOR TEST SUITE")
        print("=" * 50)
        print()

        tests = [
            self.test_state_location_query_class,
            self.test_change_location_query_class,
            self.test_state_text_query_class,
            self.test_dashboard_search_by_location,
            self.test_dashboard_search_change_patterns,
            self.test_dashboard_search_by_text,
            self.test_state_location_result_class,
            self.test_change_location_result_class,
        ]

        results = []
        for test in tests:
            try:
                passed = test()
                results.append(passed)
            except Exception as e:
                print(f"❌ Test crashed: {e}\n")
                results.append(False)

        # Summary
        print("=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        passed = sum(results)
        total = len(results)
        print(f"Passed: {passed}/{total}")

        if passed == total:
            print("✅ ALL TESTS PASSED!")
            return 0
        else:
            print(f"❌ {total - passed} test(s) failed")
            return 1


def main():
    """Run test suite."""
    try:
        suite = TestRefactor()
        return suite.run_all_tests()
    except Exception as e:
        print(f"❌ Test suite initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
