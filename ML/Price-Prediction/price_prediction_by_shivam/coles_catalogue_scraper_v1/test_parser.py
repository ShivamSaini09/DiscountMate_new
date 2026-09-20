import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from catalogue_scraper import (
    OcrLine,
    Product,
    choose_recovered_special,
    catalogue_output_base,
    classify_promo,
    deduplicate_products,
    displayed_special_price,
    excluded_offer_reason,
    excluded_page_reason,
    extract_pack_size,
    infer_region,
    invalid_product_reason,
    load_catalogue_urls,
    money_after,
    nearby_product_text,
    pair_offer_lines,
    page_numbers,
    parse_catalogue_url,
    price_from_unit_text,
    promotion_marker_kind,
    reconcile_special_price,
    trusted_products,
    unit_price_estimate,
    usable_products,
    validate_product,
)


class ParserTests(unittest.TestCase):
    def test_catalogue_url(self):
        store, slug = parse_catalogue_url(
            "https://www.catalogueau.com/coles/#catalogue=coles/example-vic-metro/"
        )
        self.assertEqual(store, "coles")
        self.assertEqual(slug, "example-vic-metro")

    def test_load_catalogue_urls_from_csv_deduplicates_by_catalogue(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "links.csv"
            path.write_text(
                "title,viewer_url\n"
                "one,https://www.catalogueau.com/coles/#catalogue=coles/first/\n"
                "duplicate,https://www.catalogueau.com/other/#catalogue=coles/first/\n"
                "two,https://www.catalogueau.com/coles/#catalogue=coles/second/\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_catalogue_urls(path),
                [
                    "https://www.catalogueau.com/coles/#catalogue=coles/first/",
                    "https://www.catalogueau.com/coles/#catalogue=coles/second/",
                ],
            )

    def test_batch_catalogue_output_is_isolated(self):
        self.assertEqual(
            catalogue_output_base(Path("results"), "coles", "weekly-vic"),
            Path("results/catalogues/coles__weekly-vic"),
        )

    def test_ocr_money_variants(self):
        self.assertEqual(money_after("SAVE", "SAVE$2.40"), 2.40)
        self.assertEqual(money_after("SAVE", "SAVES4.75"), 4.75)
        self.assertEqual(money_after("WAS", "WASS9.50"), 9.50)
        self.assertEqual(money_after("WAS", "WASSI750"), 17.50)
        self.assertEqual(money_after("WAS", "WASSI6"), 16.00)
        self.assertEqual(money_after("WAS", "WASS1I.50"), 11.50)
        self.assertEqual(money_after("SAVE", "SAVESS.75"), 5.75)
        self.assertEqual(money_after("SAVE", "SAVES]2.50"), 12.50)
        self.assertEqual(money_after("WAS", "WAS$3750"), 37.50)

    def test_product_words_are_not_offer_labels(self):
        save = OcrLine("SAVE$24.50", 90, 75, 540, 175, 558)
        dishwasher = OcrLine(
            "Finish Ultimate Dishwashing Tablets", 96, 58, 595, 336, 611
        )
        lifesavers = OcrLine("Life Savers Lollies", 96, 400, 595, 560, 611)
        self.assertEqual(pair_offer_lines([save, dishwasher, lifesavers]), [(save, None)])

    def test_region(self):
        self.assertEqual(infer_region("Coles Catalogue VIC METRO"), "VIC METRO")

    def test_page_ranges(self):
        self.assertEqual(page_numbers("2-4,7", 10), [2, 3, 4, 7])

    def test_name_keeps_text_before_unit_price(self):
        save = OcrLine("SAVE$2", 90, 90, 500, 170, 515)
        was = OcrLine("WAS$4", 90, 100, 518, 160, 532)
        lines = [
            save,
            was,
            OcrLine("Coca-Cola, Fanta or Sprite Soft", 95, 55, 542, 285, 555),
            OcrLine(
                "Drink 1.25 Litre $1.60 per litre", 95, 55, 559, 280, 574
            ),
        ]
        name, nearby, _confidence = nearby_product_text(save, was, lines, 960)
        self.assertEqual(
            name, "Coca-Cola, Fanta or Sprite Soft Drink 1.25 Litre"
        )
        self.assertIn("$1.60 per litre", nearby)

    def test_name_continues_to_third_line_after_or(self):
        save = OcrLine("SAVE$0.50", 90, 90, 500, 170, 515)
        was = OcrLine("WAS$5", 90, 100, 518, 160, 532)
        lines = [
            save,
            was,
            OcrLine("Coles Bakery Stone Baked by", 95, 55, 542, 280, 555),
            OcrLine("Laurent Sourdough Vienna or", 95, 55, 559, 280, 574),
            OcrLine("Pane di Casa $4.50 per each", 95, 55, 576, 280, 591),
        ]
        name, _nearby, _confidence = nearby_product_text(save, was, lines, 960)
        self.assertEqual(
            name,
            "Coles Bakery Stone Baked by Laurent Sourdough Vienna or Pane di Casa",
        )

    def test_name_keeps_fourth_bold_pack_line_and_normalises_ocr_units(self):
        save = OcrLine("SAVE$18", 90, 90, 500, 170, 515)
        was = OcrLine("WAS$36", 90, 100, 518, 160, 532)
        confirmed = {"source_count": 2, "variant_agreement": 1.0}
        lines = [
            save,
            was,
            OcrLine("Nature's Own Glucosamine", 95, 55, 542, 285, 555,
                    **confirmed),
            OcrLine("Sulfate With Chondroitin", 95, 55, 559, 285, 574,
                    **confirmed),
            OcrLine("Tablets 200 Pack or Vitamin D3", 95, 55, 576, 300, 591,
                    **confirmed),
            OcrLine("1000IU Capsules 400 Pack", 95, 55, 593, 285, 608,
                    **confirmed),
            OcrLine("Excludes clearance items", 95, 55, 610, 285, 625,
                    **confirmed),
        ]
        name, _nearby, confidence = nearby_product_text(
            save, was, lines, 960
        )
        self.assertEqual(
            name,
            "Nature's Own Glucosamine Sulfate With Chondroitin Tablets "
            "200 Pack or Vitamin D3 1000IU Capsules 400 Pack",
        )
        self.assertGreaterEqual(confidence, 80)

        unit_name, _nearby, _confidence = nearby_product_text(
            save,
            was,
            [
                save,
                was,
                OcrLine("Pacific West Cocktail Spring Rolls Ikg", 95,
                        55, 542, 300, 557, **confirmed),
            ],
            960,
        )
        self.assertEqual(
            unit_name, "Pacific West Cocktail Spring Rolls 1kg"
        )

    def test_name_needs_two_ocr_variants_for_trusted_confidence(self):
        anchor = OcrLine("SAVE$1", 90, 90, 500, 170, 515)
        line = OcrLine(
            "Example Product 500g", 96, 55, 542, 285, 555,
            source_count=1,
        )
        _name, _nearby, confidence = nearby_product_text(
            anchor, None, [anchor, line], 960
        )
        self.assertEqual(confidence, 79.0)

    def test_litre_and_multipack_sizes(self):
        self.assertEqual(extract_pack_size("Ice Cream Tub 2 Litre"), "2 Litre")
        self.assertEqual(extract_pack_size("Soft Drink 15x250mL"), "15x250mL")

    def test_displayed_price_combines_dollars_and_cents(self):
        was = OcrLine("WAS$2.50", 80, 96, 589, 156, 602)
        lines = [
            OcrLine('"1?', 50, 88, 504, 161, 554),
            OcrLine("25", 96, 131, 507, 161, 528),
            OcrLine("ea", 90, 132, 546, 147, 554),
            was,
        ]
        self.assertEqual(displayed_special_price(was, lines, 2.50), 1.25)

    def test_everyday_marker_and_split_price_recovery(self):
        marker = OcrLine("EVERY DAY", 95, 35, 1080, 160, 1128)
        heading = OcrLine("EVERY DAY", 95, 35, 80, 540, 170)
        self.assertEqual(promotion_marker_kind(marker), "everyday")
        self.assertEqual(promotion_marker_kind(heading), "")
        self.assertEqual(
            choose_recovered_special(
                80.0,
                "Coles Ultra Multipurpose Domestic Cleaning Wipes 50 Pack",
                "$0.10 per each",
            ),
            4.80,
        )
        self.assertEqual(
            choose_recovered_special(
                30.0, "Coles Mixed Pack Pops 825mL", "$0.88 per 100mL"
            ),
            7.30,
        )
        self.assertEqual(
            choose_recovered_special(
                450.0, "Glad Kitchen Tidy Bags 20 Pack", "$0.23 per each"
            ),
            4.50,
        )
        self.assertEqual(
            choose_recovered_special(
                14.14, "Coles Australian Almonds 750g", "$18.67 per kg"
            ),
            14.00,
        )
        self.assertEqual(
            choose_recovered_special(
                6.0, "Coles Kitchen Coleslaw", "$7.50 per kg"
            ),
            6.00,
        )

    def test_price_can_be_checked_from_unit_text(self):
        self.assertEqual(
            price_from_unit_text(
                "Schweppes Mixers or Soft Drink 1.1 Litre",
                "Drink 1.1 Litre $1.36 per litre",
            ),
            1.50,
        )
        self.assertEqual(
            price_from_unit_text(
                "Natural Chip Co Potato Chips 175g", "$2.29 per 100g"
            ),
            4.00,
        )
        self.assertEqual(
            price_from_unit_text("Mission Wraps 8 Pack 567g", "$0.79 per 100g"),
            4.50,
        )
        self.assertEqual(
            price_from_unit_text("Coles Bakery Pastries 2 Pack", "$1.50 per each"),
            3.00,
        )
        self.assertEqual(
            price_from_unit_text("Pacific West Cocktail Spring Rolls Ikg", "$0.75 per 100g"),
            7.50,
        )
        self.assertEqual(
            price_from_unit_text("Primo Cocktail Frankfurts 1k", "$6.90 per kg"),
            6.90,
        )
        self.assertEqual(
            price_from_unit_text("Don Deli Ham", "$19.90 per kg"),
            19.90,
        )
        estimate = unit_price_estimate(
            "Hercules Sandwich Bags 100 Pack", "$0.05 per each"
        )
        self.assertIsNotNone(estimate)
        self.assertGreater(estimate[1], 0.06)

    def test_unit_price_reconstructs_multipacks_and_odd_cents(self):
        self.assertEqual(
            price_from_unit_text("Danone Ultimate Yoghurt 4x115g", "$0.91 per 100g"),
            4.20,
        )
        self.assertEqual(
            price_from_unit_text("Pedigree Dentastix 28 Pack", "$0.60 per each"),
            16.80,
        )
        self.assertEqual(
            price_from_unit_text(
                "Paper Cups 250mL 10 Pack or Paper Plates 10 Pack",
                "$0.30 per each",
            ),
            3.00,
        )
        self.assertEqual(
            price_from_unit_text(
                "Mt Franklin Lightly Sparkling Water 1.25 Litre",
                "$1.30 per litre",
            ),
            1.62,
        )
        self.assertEqual(
            price_from_unit_text(
                "Ostelin Calcium & Vitamin D3 + K2 Tablets 60 Pack",
                "$34.17 per 100 each",
            ),
            20.50,
        )

    def test_wrong_headline_price_is_reconciled_from_unit_price(self):
        self.assertEqual(
            reconcile_special_price(
                95.0, None,
                "Oral B Electric Toothbrush 1 Pack", "$50.00 per each",
            ),
            50.0,
        )
        self.assertEqual(
            reconcile_special_price(
                5.0, None, "Tassal Smoked Salmon 250g", "$60.00 per kg",
            ),
            15.0,
        )
        self.assertEqual(
            reconcile_special_price(
                10.0, 10.0, "Vaalia Yoghurt Pouch 140g", "$1.43 per 100g",
                is_multibuy=True,
            ),
            10.0,
        )

    def test_near_half_discount_is_not_misclassified(self):
        self.assertEqual(
            classify_promo(6.60, 12.60, "WAS $12.60"), "save_amount"
        )
        self.assertEqual(
            classify_promo(6.30, 12.60, "WAS $12.60"), "half_price"
        )
        self.assertEqual(
            classify_promo(6.60, 12.60, "1/2 PRICE"), "half_price"
        )

    def test_trusted_price_requires_two_independent_signals(self):
        one_signal = Product(
            retailer="Coles", region="VIC", catalogue_title="Catalogue",
            catalogue_start_date="", catalogue_end_date="", page_number=5,
            product_name="Example Drink 1 Litre", special_price=4.0,
            regular_price=5.0, save_amount=1.0, discount_percent=20.0,
            promo_type="save_amount", pack_size="1 Litre", unit_price_text="",
            offer_text="SAVE$1 | WAS$5 | Example Drink 1 Litre",
            ocr_confidence=95.0, needs_review=False, review_reason="",
            source_page_image="", source_catalogue_url="",
        )
        validate_product(one_signal)
        self.assertTrue(one_signal.needs_review)
        self.assertEqual(one_signal.verification_status, "review")
        self.assertIn("single_price_evidence", one_signal.review_reason)
        self.assertEqual(trusted_products([one_signal]), [])

        two_signals = replace(
            one_signal,
            unit_price_text="$4.00 per litre",
            offer_text=(
                "SAVE$1 | WAS$5 | Example Drink 1 Litre | $4.00 per litre"
            ),
            needs_review=False,
            review_reason="",
            price_source="",
            price_evidence="",
            evidence_count=0,
            verification_status="unvalidated",
        )
        validate_product(two_signals)
        self.assertFalse(two_signals.needs_review)
        self.assertEqual(two_signals.verification_status, "verified")
        self.assertEqual(two_signals.evidence_count, 2)
        self.assertEqual(trusted_products([two_signals]), [two_signals])

    def test_unit_price_conflict_is_never_trusted(self):
        product = Product(
            retailer="Coles", region="VIC", catalogue_title="Catalogue",
            catalogue_start_date="", catalogue_end_date="", page_number=5,
            product_name="Example Drink 1 Litre", special_price=14.0,
            regular_price=15.0, save_amount=1.0, discount_percent=6.67,
            promo_type="save_amount", pack_size="1 Litre",
            unit_price_text="$4.00 per litre",
            offer_text=(
                "SAVE$1 | WAS$15 | Example Drink 1 Litre | $4.00 per litre"
            ),
            ocr_confidence=95.0, needs_review=False, review_reason="",
            source_page_image="", source_catalogue_url="",
            price_evidence="displayed_price",
        )
        validate_product(product)
        self.assertTrue(product.needs_review)
        self.assertIn("unit_price_conflict", product.review_reason)
        self.assertEqual(product.verification_status, "review")

    def test_invalid_prices_are_dropped_but_unit_conflicts_are_kept(self):
        valid = Product(
            retailer="Coles", region="VIC", catalogue_title="Catalogue",
            catalogue_start_date="", catalogue_end_date="", page_number=1,
            product_name="Multi-buy Drink 1 Litre", special_price=10.0,
            regular_price=12.0, save_amount=2.0, discount_percent=16.67,
            promo_type="save_amount", pack_size="1 Litre",
            unit_price_text="$5.00 per litre", offer_text="$5.00 per litre",
            ocr_confidence=95.0, needs_review=True,
            review_reason="unit_price_conflict", source_page_image="",
            source_catalogue_url="", verification_status="review",
        )
        missing = replace(valid, special_price=None)
        wrong_order = replace(valid, special_price=13.0)
        broken_math = replace(valid, special_price=9.0)

        self.assertEqual(invalid_product_reason(valid), "")
        self.assertEqual(
            invalid_product_reason(missing), "special_price_missing"
        )
        self.assertEqual(
            invalid_product_reason(wrong_order), "special_not_below_regular"
        )
        self.assertEqual(
            invalid_product_reason(broken_math), "price_arithmetic_conflict"
        )
        self.assertEqual(
            usable_products([valid, missing, wrong_order, broken_math]), [valid]
        )

    def test_duplicate_products_keep_complete_price_row(self):
        base = Product(
            retailer="Coles", region="VIC", catalogue_title="Catalogue",
            catalogue_start_date="", catalogue_end_date="", page_number=5,
            product_name="Example Drink 1 Litre", special_price=4.0,
            regular_price=5.0, save_amount=1.0, discount_percent=20.0,
            promo_type="save_amount", pack_size="1 Litre", unit_price_text="",
            offer_text="", ocr_confidence=90.0, needs_review=False,
            review_reason="", source_page_image="", source_catalogue_url="",
        )
        incomplete = replace(
            base, regular_price=None, save_amount=None, needs_review=True,
            review_reason="save_amount_missing;regular_price_missing",
            ocr_confidence=96.0,
        )
        self.assertEqual(deduplicate_products([incomplete, base]), [base])

    def test_non_grocery_promotional_pages_are_excluded(self):
        self.assertEqual(
            excluded_page_reason(
                [OcrLine("LIQUORLAND", 95, 100, 70, 800, 160)]
            ),
            "liquorland_page",
        )
        self.assertEqual(
            excluded_page_reason(
                [OcrLine("colesmobile", 95, 80, 80, 300, 140)]
            ),
            "mobile_sim_page",
        )
        self.assertEqual(
            excluded_page_reason(
                [OcrLine("GIFT CARDS", 95, 100, 80, 400, 150)]
            ),
            "gift_card_page",
        )

    def test_sim_and_gift_card_offers_are_excluded(self):
        self.assertEqual(
            excluded_offer_reason(
                "Optus",
                "WAS$59 | WAS$59 | Optus $59 Prepaid SIM | 85GB data",
            ),
            "sim_offer",
        )
        self.assertEqual(
            excluded_offer_reason(
                "Apple Gift Card", "SAVE$10 | Apple Gift Card $50"
            ),
            "gift_card_offer",
        )

    def test_phone_hardware_and_footer_terms_are_retained(self):
        self.assertEqual(
            excluded_offer_reason(
                "Optus X Pro 5G + 6.52 inch display",
                "WAS$169 | Optus X Pro 5G | Includes SIM Starter Kit",
            ),
            "",
        )
        self.assertEqual(
            excluded_offer_reason(
                "Everyday Grocery Product",
                "SAVE$2 | WAS$4 | Everyday Grocery Product | Excludes gift cards",
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
