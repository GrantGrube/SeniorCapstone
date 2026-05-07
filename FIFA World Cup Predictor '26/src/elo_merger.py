import pandas as pd

# Load datasets
elo = pd.read_csv("FIFA World Cup Predictor '26/Data/elo_ratings.csv", header=None, names=["rank", "abbreviation", "elo_rating"])
full = pd.read_csv("FIFA World Cup Predictor '26/Data/Processed/full_merged_dataset.csv")

# Map abbreviation to Country name
abbrev_to_country = {
    'ES': 'Spain',                          'AR': 'Argentina',
    'FR': 'France',                         'EN': 'England',
    'CO': 'Colombia',                       'BR': 'Brazil',
    'PT': 'Portugal',                       'NL': 'Netherlands',
    'EC': 'Ecuador',                        'HR': 'Croatia',
    'NO': 'Norway',                         'DE': 'Germany',
    'CH': 'Switzerland',                    'UY': 'Uruguay',
    'TR': 'Turkey',                         'JP': 'Japan',
    'SN': 'Senegal',                        'DK': 'Denmark',
    'IT': 'Italy',                          'MX': 'Mexico',
    'BE': 'Belgium',                        'PY': 'Paraguay',
    'AT': 'Austria',                        'MA': 'Morocco',
    'CA': 'Canada',                         'UA': 'Ukraine',
    'SQ': 'Albania',                        'KR': 'South Korea',
    'RU': 'Russia',                         'AU': 'Australia',
    'RS': 'Serbia',                         'GR': 'Greece',
    'IR': 'Iran',                           'US': 'United States',
    'NG': 'Nigeria',                        'PL': 'Poland',
    'PA': 'Panama',                         'CZ': 'Czech Republic',
    'CL': 'Chile',                          'DZ': 'Algeria',
    'UZ': 'Uzbekistan',                     'WA': 'Wales',
    'VE': 'Venezuela',                      'KO': 'Kosovo',
    'PE': 'Peru',                           'HU': 'Hungary',
    'SI': 'Slovenia',                       'JO': 'Jordan',
    'IE': 'Republic of Ireland',            'SK': 'Slovakia',
    'BO': 'Bolivia',                        'SE': 'Sweden',
    'EG': 'Egypt',                          'GE': 'Georgia',
    'RO': 'Romania',                        'CD': 'DR Congo',
    'CI': 'Ivory Coast',                    'CR': 'Costa Rica',
    'IL': 'Israel',                         'TN': 'Tunisia',
    'CM': 'Cameroon',                       'EI': 'Republic of Ireland',
    'NM': 'North Macedonia',               'SA': 'Saudi Arabia',
    'ML': 'Mali',                           'NZ': 'New Zealand',
    'IQ': 'Iraq',                           'BA': 'Bosnia and Herzegovina',
    'HN': 'Honduras',                       'IS': 'Iceland',
    'CV': 'Cape Verde',                     'AO': 'Angola',
    'JM': 'Jamaica',                        'HT': 'Haiti',
    'AE': 'United Arab Emirates',          'BF': 'Burkina Faso',
    'ZA': 'South Africa',                  'GT': 'Guatemala',
    'GH': 'Ghana',                          'FI': 'Finland',
    'BY': 'Belarus',                        'OM': 'Oman',
    'GN': 'Guinea',                         'SY': 'Syria',
    'PS': 'Palestine',                      'CW': 'Curaçao',
    'NS': 'Northern Ireland',              'BG': 'Bulgaria',
    'ME': 'Montenegro',                     'SR': 'Suriname',
    'LY': 'Libya',                          'QA': 'Qatar',
    'GM': 'Gambia',                         'KD': 'Kurdistan',
    'BH': 'Bahrain',                        'KZ': 'Kazakhstan',
    'CN': 'China',                          'BJ': 'Benin',
    'GA': 'Gabon',                          'NE': 'Niger',
    'TT': 'Trinidad and Tobago',           'LU': 'Luxembourg',
    'UG': 'Uganda',                         'AM': 'Armenia',
    'GQ': 'Equatorial Guinea',             'FO': 'Faroe Islands',
    'KP': 'North Korea',                    'KM': 'Comoros',
    'MZ': 'Mozambique',                     'ZM': 'Zambia',
    'MG': 'Madagascar',                     'TH': 'Thailand',
    'EE': 'Estonia',                        'SD': 'Sudan',
    'RE': 'Réunion',                        'SL': 'Sierra Leone',
    'KE': 'Kenya',                          'ZW': 'Zimbabwe',
    'TG': 'Togo',                           'ID': 'Indonesia',
    'TZ': 'Tanzania',                       'AZ': 'Azerbaijan',
    'GP': 'Guadeloupe',                     'LB': 'Lebanon',
    'MQ': 'Martinique',                     'VN': 'Vietnam',
    'KW': 'Kuwait',                         'NI': 'Nicaragua',
    'SV': 'El Salvador',                    'MY': 'Malaysia',
    'KG': 'Kyrgyzstan',                     'MR': 'Mauritania',
    'RW': 'Rwanda',                         'CY': 'Cyprus',
    'LR': 'Liberia',                        'TJ': 'Tajikistan',
    'NC': 'New Caledonia',                 'DO': 'Dominican Republic',
    'MD': 'Moldova',                        'LV': 'Latvia',
    'BW': 'Botswana',                       'MT': 'Malta',
    'LT': 'Lithuania',                      'GY': 'Guyana',
    'ET': 'Ethiopia',                       'MW': 'Malawi',
    'GW': 'Guinea-Bissau',                 'BI': 'Burundi',
    'CF': 'Central African Republic',      'CU': 'Cuba',
    'GF': 'French Guiana',                 'TM': 'Turkmenistan',
    'LS': 'Lesotho',                        'SW': 'Eswatini',
    'CG': 'Congo',                          'TI': 'Timor-Leste',
    'VC': 'Saint Vincent and the Grenadines', 'PH': 'Philippines',
    'YE': 'Yemen',                          'ER': 'Eritrea',
    'HK': 'Hong Kong',                      'PG': 'Papua New Guinea',
    'SS': 'South Sudan',                    'PR': 'Puerto Rico',
    'SG': 'Singapore',                      'IN': 'India',
    'TD': 'Chad',                           'GD': 'Grenada',
    'BM': 'Bermuda',                        'VU': 'Vanuatu',
    'FJ': 'Fiji',                           'BZ': 'Belize',
    'MU': 'Mauritius',                      'SB': 'Solomon Islands',
    'AD': 'Andorra',                        'ST': 'São Tomé and Príncipe',
    'AF': 'Afghanistan',                    'GI': 'Gibraltar',
    'LC': 'Saint Lucia',                    'KN': 'Saint Kitts and Nevis',
    'JS': 'Jersey',                         'SO': 'Somalia',
    'AW': 'Aruba',                          'GL': 'Greenland',
    'MM': 'Myanmar',                        'BD': 'Bangladesh',
    'DM': 'Dominica',                       'BB': 'Barbados',
    'NP': 'Nepal',                          'DJ': 'Djibouti',
    'MC': 'Monaco',                         'LI': 'Liechtenstein',
    'AG': 'Antigua and Barbuda',           'TW': 'Taiwan',
    'KH': 'Cambodia',                       'SC': 'Seychelles',
    'MV': 'Maldives',                       'SM': 'San Marino',
    'PK': 'Pakistan',                       'KY': 'Cayman Islands',
    'LK': 'Sri Lanka',                      'HG': 'Guernsey',
    'MN': 'Mongolia',                       'WS': 'Samoa',
    'BS': 'Bahamas',                        'GU': 'Guam',
    'LA': 'Laos',                           'VG': 'British Virgin Islands',
    'VA': 'Vatican City',                   'AB': 'Abkhazia',
    'TL': 'East Timor',                     'BN': 'Brunei',
    'AI': 'Anguilla',                       'CK': 'Cook Islands',
    'BT': 'Bhutan',                         'FM': 'Micronesia',
    'MH': 'Marshall Islands',              'KI': 'Kiribati',
    'TO': 'Tonga',                          'NU': 'Niue',
    'PW': 'Palau',                          'AS': 'American Samoa',
    'YT': 'Mayotte',                        'AL': 'Algeria',
    'ZN': 'Zimbabwe',                       'EU': 'European Union',
    'MF': 'Saint Martin',                   'EH': 'Western Sahara',
    'MS': 'Montserrat',                     'SX': 'Sint Maarten',
    'TV': 'Tuvalu',                         'BL': 'Saint Barthélemy',
    'PM': 'Saint Pierre and Miquelon',      'TC': 'Turks and Caicos Islands',
    'MO': 'Macau',                          'FK': 'Falkland Islands',
    'MP': 'Northern Mariana Islands',
}

# Add country_name column to ELO dataframe
elo["country_name"] = elo["abbreviation"].map(abbrev_to_country)

# Warn about any unmapped abbreviations
unmapped = elo[elo["country_name"].isna() & elo["abbreviation"].notna()]
if not unmapped.empty:
    print(f"WARNING: {len(unmapped)} ELO abbreviation(s) have no country mapping:")
    print(unmapped[["abbreviation"]].to_string(index=False))

# Merge ELO ratings into the full dataset
# We merge twice for home_team and away_team.

elo_lookup = elo[["country_name", "elo_rating", "rank"]].rename(columns={"country_name": "team"})

full = full.merge(
    elo_lookup.rename(columns={"team": "home_team", "elo_rating": "home_elo", "rank": "home_elo_rank"}),
    on="home_team",
    how="left"
)

full = full.merge(
    elo_lookup.rename(columns={"team": "away_team", "elo_rating": "away_elo", "rank": "away_elo_rank"}),
    on="away_team",
    how="left"
)

# Summary

total = len(full)
home_matched = full["home_elo"].notna().sum()
away_matched = full["away_elo"].notna().sum()

print(f"\nMerge complete!")
print(f"  Total rows: {total:,}")
print(f"  home_elo matched: {home_matched:,} ({home_matched/total*100:.1f}%)")
print(f"  away_elo matched: {away_matched:,} ({away_matched/total*100:.1f}%)")
print(f"\nSample output:")
print(full[["home_team", "home_elo", "home_elo_rank", "away_team", "away_elo", "away_elo_rank"]].head(10).to_string(index=False))

# Save output

full.to_csv("full_merged_dataset.csv", index=False)
elo.to_csv("elo_ratings_with_names.csv", index=False)

print("\nFiles saved:")
print("  full_merged_dataset.csv — main dataset with home_elo, away_elo columns")
print("  elo_ratings_with_names.csv — ELO table with country names added")