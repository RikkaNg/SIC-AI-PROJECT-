# -*- coding: utf-8 -*-
"""
generate_product_names.py (v1.0)
Sinh tên sản phẩm cụ thể (kiểu brand quốc tế, phong cách Favorita) cho toàn bộ
items trong items.csv — dữ liệu Kaggle gốc không có cột tên.

- Deterministic: cùng item_nbr/class luôn ra cùng tên (seed từ item_nbr).
- Cấu trúc: BRAND + LOẠI SP + BIẾN THỂ (flavor/style) + DUNG TÍCH/ĐÓNG GÓI.
- Bảo đảm unique toàn cục 4,100 tên; nếu trùng -> ghém thêm biến thể phân biệt.

Chạy: python ml_training/src/generate_product_names.py
Xuất: ml_training/data/raw/product_names.csv (item_nbr, name)
"""

import sys
import random
from pathlib import Path

import pandas as pd

# Console Windows mặc định cp1252 - ép UTF-8 để log tiếng Việt không crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
PROJECT_ROOT = BASE_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "product_names.csv"

# ====================== WORD BANK THEO FAMILY ======================
# Mỗi family: brands (thương hiệu), bases (loại sản phẩm), variants (biến thể),
# sizes (đóng gói/dung tích). Tên ghép: BRAND + BASE + VARIANT + SIZE.
FAMILY_WORDS: dict = {
    "BEVERAGES": {
        "brands": ["GATORED", "FIZZO", "AQUAPURA", "CITRA SPLASH", "TROPILUX", "SODAVIA", "FRUTIGO", "VITALIFT"],
        "bases": ["SPORTS DRINK", "SOFT DRINK", "SPARKLING WATER", "FRUIT JUICE", "ICED TEA", "ENERGY DRINK",
                  "ORANGE SODA", "COLA", "LEMONADE", "APPLE JUICE", "FRUIT PUNCH", "MINERAL WATER"],
        "variants": ["POME-BERRY", "TROPICAL MANGO", "CLASSIC LEMON", "ORANGE BLAST", "STRAWBERRY KIWI",
                     "GUAVA NECTAR", "PEACH PASSION", "LIME COOLER", "GRAPE FIZZ", "PINEAPPLE CRUSH", "ZERO SUGAR"],
        "sizes": ["20OZ", "1.5L", "500ML", "2L", "330ML", "6PK 250ML"],
    },
    "DAIRY": {
        "brands": ["LACTEAL", "VIDALECHE", "CREMAX", "NATUFRESH", "MONTEROSA", "YOGURINO", "QUESARIA DEL VALLE"],
        "bases": ["WHOLE MILK", "YOGURT", "FRESH CHEESE", "AGED CHEESE", "BUTTER", "CREAM", "SKIM MILK",
                  "DRINKABLE YOGURT", "MOZZARELLA", "SOUR CREAM", "CONDENSED MILK"],
        "variants": ["ENTERO", "LIGHT", "VANILLA", "NATURAL", "STRAWBERRY", "MANCHEGO STYLE", "GREEK STYLE",
                     "FRESH FARM", "LOW FAT", "PROBIOTIC"],
        "sizes": ["1L", "500G", "250G", "946ML", "120G", "400G"],
    },
    "PRODUCE": {
        "brands": ["FRESCO VALLE", "CAMPO VERDE", "ORGANICA", "HUERTA DEL SOL", "COSTA FRUTA", "NATIVO"],
        "bases": ["BANANOS", "PLANTANOS", "TOMATES", "CEBOLLA", "PAPAYA", "MANZANA", "NARANJA", "LECHUGA",
                  "ZANAHORIA", "PAPA", "LIMONES", "AGUACATE", "FRESAS", "BROCOLI", "MELON"],
        "variants": ["FRESCO", "ORGANICO", "PREMIUM SELECT", "EXTRA DULCE", "PRIMERA CALIDAD", "LOCAL", "EXPORT"],
        "sizes": ["1KG", "500G", "2KG", "UNIDAD", "LIBRA", "MANOJO"],
    },
    "GROCERY I": {
        "brands": ["DORADA", "LA MOLINERA", "CASERA PLUS", "EL SOL", "AZUCARINA", "GRANOS DEL ECUADOR"],
        "bases": ["RICE", "SUGAR", "PASTA SPAGHETTI", "COOKING OIL", "SALT", "LENTILS", "FLOUR", "BLACK BEANS",
                  "TUNA IN OIL", "TOMATO SAUCE", "OATMEAL", "CORN MEAL", "MAYONNAISE"],
        "variants": ["EXTRA VIRGIN", "TIPO 1", "PREMIUM", "CLASICA", "REFINADA", "INTEGRAL", "CON SAL",
                     "EN TROZOS", "IMPORTADA", "FAMILY PACK"],
        "sizes": ["1KG", "2KG", "5KG", "900G", "470ML", "160G"],
    },
    "GROCERY II": {
        "brands": ["AMBAR", "COCINA real".upper(), "SABORIX", "LA PENINSULAR", "DULCESUR", "ALIMENTA"],
        "bases": ["COFFEE GROUND", "CANNED PEACHES", "PEANUT BUTTER", "CORN FLAKES", "CHOCOLATE POWDER",
                  "SARDINES", "GREEN PEAS", "MUSHROOMS", "JELLY POWDER", "PANCAKE MIX", "HONEY"],
        "variants": ["INSTANT", "EN ALMIBAR", "CREAMY", "CRUNCHY", "SIN AZUCAR", "EXTRA DARK", "SUAVE",
                     "TRADICIONAL", "DULCE", "AHUMADO"],
        "sizes": ["200G", "400G", "820G", "340G", "150G", "1KG"],
    },
    "CLEANING": {
        "brands": ["LIMPIOL", "BRILLANTE", "ALCOLUX", "FRESHIA", "DOMADOR", "HIGIENA PLUS"],
        "bases": ["LIQUID DETERGENT", "FABRIC SOFTENER", "DISINFECTANT", "DISH SOAP", "BLEACH", "FLOOR CLEANER",
                  "GLASS CLEANER", "MULTIPURPOSE SPRAY", "BAR SOAP", "SCRUB SPONGE"],
        "variants": ["LAVANDA", "MANZANA", "PINO", "CITRUS", "ANTIBACTERIAL", "CONCENTRADO", "DOUBLE POWER",
                     "SIN PERFUME", "MARSELLA", "ACTIVE FRESH"],
        "sizes": ["1L", "3L", "900ML", "500ML", "2L", "3CT"],
    },
    "PERSONAL CARE": {
        "brands": ["DERMALIS", "ESENCIA", "CUIDADO PLUS", "SENSUEL", "FRESHMAN", "BELLA PIEL"],
        "bases": ["SHAMPOO", "CONDITIONER", "BODY WASH", "TOOTHPASTE", "DEODORANT", "HAND CREAM",
                  "FACIAL SOAP", "SHAVING CREAM", "BABY LOTION", "MOUTHWASH"],
        "variants": ["CABELLO SECO", "MENTOL", "ALOE VERA", "MANZANA", "FOR MEN", "SENSITIVE", "ULTRA HIDRATANTE",
                     "CARBON ACTIVADO", "FLUOR TOTAL", "ROLL-ON"],
        "sizes": ["400ML", "90G", "150ML", "750ML", "200ML", "50ML"],
    },
    "HOME CARE": {
        "brands": ["HOGARLIN", "ECOVELA", "VELADORA", "FIESTA BOWL", "PRACTIPLUS", "TALLER CASA"],
        "bases": ["CANDLES", "PAPER TOWELS", "TOILET PAPER", "TRASH BAGS", "MATCHES", "MOUSETRAP",
                  "AIR FRESHENER", "LIGHT BULB", "BATTERIES", "INSECT SPRAY"],
        "variants": ["VANILLA SCENT", "DOBLE HOJA", "JUMBO ROLL", "RESISTENTE", "LED WARM", "AAA 4PK",
                     "AA 8PK", "MULTI INSECTO", "OLEO", "GARLIC SCENT"],
        "sizes": ["12CT", "4 ROLLS", "1PK", "10CT", "60W", "300ML"],
    },
    "BEAUTY": {
        "brands": ["LUMINA", "ROSA FINA", "GLAMOUR ECUADOR", "BELLEZA MAX", "SILUETA", "VITANAIL"],
        "bases": ["LIPSTICK", "NAIL POLISH", "FACE POWDER", "MASCARA", "PERFUME", "EYE SHADOW PALETTE",
                  "HAIR GEL", "BB CREAM", "BLUSH", "MAKEUP REMOVER"],
        "variants": ["ROJO CLASICO", "MATTE", "PEARL SHINE", "VOLUMEN EXTREMO", "FLORAL NOIR", "TONOS CALIDOS",
                     "FIX EXTREMO", "TONO MEDIO", "WATERPROOF", "MINERAL"],
        "sizes": ["15ML", "8ML", "30G", "50ML", "100ML", "1UN"],
    },
    "BREAD/BAKERY": {
        "brands": ["PANADERIA ORO", "HORNO SANTO", "MIGA DORADA", "TRIGAL", "LA ESPIGA", "DULCE HORNO"],
        "bases": ["WHITE BREAD", "WHOLE WHEAT BREAD", "SWEET BUNS", "CROISSANT", "PAN DE YUCA", "CAKE SLICE",
                  "BAGUETTE", "DONUTS", "TOAST BREAD", "MUFFINS"],
        "variants": ["INTEGRAL", "CON QUESO", "RELLENO DE MANJAR", "MANTEquilla".upper(), "ARTESANAL",
                     "SIN AZUCAR", "EXTRA SUAVE", "DE CAMPO", "MULTIGRANO", "GLASEADO"],
        "sizes": ["500G", "6CT", "12CT", "1UN", "350G", "8CT"],
    },
    "FROZEN FOODS": {
        "brands": ["POLAR CHEF", "FRIOSUR", "GLACIAR", "NORDICA", "ANTARTIDA", "FREEZMAX"],
        "bases": ["FROZEN PEAS", "FRENCH FRIES", "ICE CREAM", "FROZEN NUGGETS", "FROZEN PIZZA", "ICE POPS",
                  "FROZEN WAFFLES", "FROZEN BURGERS", "SHERBET", "FROZEN CORN"],
        "variants": ["VAINILLA", "CHOCOLATE", "DELUXE", "CONDIMENTADAS", "FAMILIAR", "MIXTAS", "CREMOSO",
                     "EXTRA CRISPY", "DE FRUTA", "QUESO"],
        "sizes": ["500G", "1KG", "750ML", "400G", "8CT", "1L"],
    },
    "MEATS": {
        "brands": ["CARNES DEL VALLE", "ASADOR PLUS", "PRADERA", "RES BRAVO", "CHARCUTERIA REAL", "GRILLMAX"],
        "bases": ["GROUND BEEF", "BEEF STEAK", "PORK CHOPS", "HAM SLICED", "SAUSAGES", "BACON",
                  "MORTADELLA", "BEEF RIBS", "SALAMI", "PORK LEG"],
        "variants": ["PRIMERA", "MOLIDO ESPECIAL", "AHUMADO", "AHUMADA", "TIPO HUNGARO", "MARINADO",
                     "MADURADO", "EXTRA MAGRO", "PARRILLERO", "EN TAJADAS"],
        "sizes": ["1KG", "500G", "300G", "250G", "2KG", "PAQUETE"],
    },
    "POULTRY": {
        "brands": ["AVICOLA REAL", "GALLINA DORADA", "POLLOSUR", "HUEVO ORO", "GRANJA SAN MIGUEL", "PLUMAVIVA"],
        "bases": ["WHOLE CHICKEN", "CHICKEN BREAST", "CHICKEN WINGS", "CHICKEN DRUMSTICKS", "GROUND CHICKEN",
                  "TURKEY SLICED", "CHICKEN LIVERS"],
        "variants": ["FRESCO", "CONGELADO", "MARMITAS", "SIN PIEL", "MARINADO", "PRIMERA", "ENTERO LIMPIO"],
        "sizes": ["1KG", "1.5KG", "2KG", "500G", "BANDEJA", "LIBRA"],
    },
    "SEAFOOD": {
        "brands": ["MAR AZUL", "PESCADOR", "ATLANTIDA", "COSTA PESCA", "OLAS DEL PACIFICO", "MAREJADA"],
        "bases": ["TUNA CHUNKS", "SHRIMP", "FISH FILLET", "SARDINES IN TOMATO", "MUSSELS", "OCTOPUS",
                  "CRAB MEAT", "SQUID RINGS", "SALMON PORTION"],
        "variants": ["EN AGUA", "AL LIMON", "PILADO", "FRESCO", "CONGELADO", "PREMIUM", "ESCURRIDO",
                     "AHUMADO", "EN SU JUGO", "MARINADO"],
        "sizes": ["160G", "1KG", "500G", "400G", "LATA", "250G"],
    },
    "DELI": {
        "brands": ["FUENTE FINA", "SABOR CASERO", "DEGUSTA", "LA TABLA", "GOURMETIX", "MESON DEL CHEF"],
        "bases": ["CHEESE PLATTER", "OLIVE MIX", "STUFFED PEPPERS", "PATE", "QUICHE", "SUSHI ROLL",
                  "STUFFED OLIVES", "ANTIPASTO", "CHICKEN SALAD", "POTATO SALAD"],
        "variants": ["PREPARADO", "ARTESANAL", "DEL DIA", "CON JAMON", "VEGETARIANO", "PICANTE",
                     "CREMOSO", "AL AJILLO", "ESPECIAL", "CLASSICO"],
        "sizes": ["250G", "300G", "150G", "4CT", "1UN", "500G"],
    },
    "EGGS": {
        "brands": ["GRANJA ORO", "HUEVOS FELICES", "AVIGAL", "CAMPO CLARO", "LA NIDADA", "RANCHO AVICOLA"],
        "bases": ["CHICKEN EGGS", "QUAIL EGGS", "EGG WHITES"],
        "variants": ["TAMANO GRANDE", "AA", "A", "DE PASTO", "OMEGA 3", "BLANCOS", "ROJOS", "FERTILES"],
        "sizes": ["12CT", "30CT", "6CT", "15CT", "36CT", "180CT"],
    },
    "PREPARED FOODS": {
        "brands": ["Listo Chef".upper(), "COMIDAS EXPRESS", "MAMITA", "SABOR HOY", "RINCONCITO", "WOK PLUS"],
        "bases": ["CHICKEN MEAL", "RICE WITH CHICKEN", "CEVICHE", "ENSENADA PLATTER", "SOUP BOWL", "PASTA PLATE",
                  "TAMALES", "EMPANADAS", "SANDWICH WRAP", "BBQ PLATE"],
        "variants": ["CASERO", "PICANTE", "CON ENSALADA", "DIARIO", "ESPECIAL", "AL VAPOR", "GRATINADO",
                     "TIPICO", "LIGHT", "FAMILIAR"],
        "sizes": ["1UN", "400G", "2CT", "500G", "PORCION", "300G"],
    },
    "LIQUOR,WINE,BEER": {
        "brands": ["ANDINA", "VINA SOLAR", "CERVEZA PILSERA", "RON DEL CARIBE", "LUJOSA", "DESTILERIA REAL"],
        "bases": ["LAGER BEER", "RED WINE", "WHITE WINE", "PILSENER", "RUM", "WHISKEY", "VODKA",
                  "CRAFT ALE", "CIDER", "ROSE WINE"],
        "variants": ["GRAN RESERVA", "EXTRA FRIA", "SECO", "DULCE", "AÑEJO", "IMPORTADO", "SIN ALCOHOL",
                     "RESERVA FAMILIAR", "TRIPLE FILTRADO", "BRUT"],
        "sizes": ["330ML", "750ML", "650ML", "1L", "6PK", "1.5L"],
    },
    "AUTOMOTIVE": {
        "brands": ["RUTERO", "MOTORPLUS", "TALLERINO", "VELOCIDAD", "AUTOMAX", "CARRERA PRO"],
        "bases": ["MOTOR OIL", "AIR FRESHENER", "CAR WAX", "BRAKE FLUID", "COOLANT", "WIPER BLADES",
                  "CAR SHAMPOO", "BATTERY TERMINAL", "FUSE KIT", "TIRE INFLATOR"],
        "variants": ["20W50", "SINTETICO", "PIÑA SCENT", "PULIDOR", "DOT 4", "REFRIGERANTE VERDE",
                     "UNIVERSAL", "TURBO", "MULTIGRADE", "PARABRISAS"],
        "sizes": ["1L", "4L", "250ML", "2CT", "1PK", "500ML"],
    },
    "BABY CARE": {
        "brands": ["PEQUESUAVE", "NANITO", "MUNECO SUAVE", "BABYVITA", "PEQUEPASOS", "CUNA DORADA"],
        "bases": ["DIAPERS", "BABY WIPES", "BABY SHAMPOO", "BABY CEREAL", "BABY FORMULA", "BABY FOOD JAR",
                  "PACIFIER", "BABY OIL", "NURSING PADS"],
        "variants": ["ETAPA 1", "ETAPA 2", "ETAPA 3", "HIDRATANTE", "SIN PERFUME", "EXTRA ABSORBENTE",
                     "MANZANA", "MULTIFRUTAS", "ULTRA SUAVE", "XG"],
        "sizes": ["40CT", "80CT", "200ML", "400G", "12CT", "1UN"],
    },
    "BOOKS": {
        "brands": ["LETRAS VIVAS", "EDITORIAL ANDINA", "LIBROTECA", "PAGINA UNO", "CULTURA PLUS", "VERBO"],
        "bases": ["NOVEL", "NOTEBOOK", "COLORING BOOK", "COOKBOOK", "CHILDREN STORYBOOK", "DICTIONARY",
                  "COMIC VOLUME", "GUIDEBOOK", "WORKBOOK", "POETRY ANTHOLOGY"],
        "variants": ["TAPA DURA", "EDICION BILINGUE", "CUADRICULADO", "VOL 1", "ILUSTRADO", "ESCOLAR",
                     "DE BOLSILLO", "ACTUALIZADO", "100 PAGINAS", "COLECCIONISTA"],
        "sizes": ["1UN", "96HOJAS", "50HOJAS", "200PAG", "A4", "A5"],
    },
    "MAGAZINES": {
        "brands": ["REVISTA HOY", "PORTADA", "ACTUALIDAD", "ORBITA", "LA SEMANA", "TITULARES"],
        "bases": ["WEEKLY MAGAZINE", "FASHION MAGAZINE", "SPORTS MAGAZINE", "TV GUIDE", "COOKING MAGAZINE",
                  "CAR MAGAZINE", "HEALTH MAGAZINE", "KIDS MAGAZINE", "TECH MAGAZINE", "CROSSWORD BOOK"],
        "variants": ["EDICION ESPECIAL", "NUMERO 1", "COLECCION", "ANUARIO", "EDICION VERANO",
                     "SORPRESA INCLUIDA", "ESPECIAL MODA", "FUTBOL", "GOURMET", "CRUCIGRAMAS"],
        "sizes": ["1UN", "MENSUAL", "SEMANAL", "12CT", "ED. NUEVA", "BIMESTRAL"],
    },
    "SCHOOL AND OFFICE SUPPLIES": {
        "brands": ["ESCOLARMAX", "PUNTAVIVA", "OFIXPRESS", "TRAZOS", "LETTERA", "CARTABON"],
        "bases": ["BALLPOINT PEN", "PENCIL SET", "ERASER", "GLUE STICK", "SCISSORS", "MARKER SET",
                  "SPIRAL NOTEBOOK", "STAPLER", "CALCULATOR", "CORRECTION TAPE", "RULER"],
        "variants": ["AZUL", "NEGRO", "DE COLORES", "PUNTO FINO", "TITANIO", "8GB", "CIENTIFICA",
                     "ADHESIVO PERMANENTE", "DOBLE CARTA", "PASTEL"],
        "sizes": ["12CT", "1UN", "3CT", "50HOJAS", "100CT", "SET"],
    },
    "HOME APPLIANCES": {
        "brands": ["ELECTROHOGAR", "VOLTAIA", "COCINA MAX", "DOMOTICA", "LUMEX", "FRIOMAX"],
        "bases": ["BLENDER", "RICE COOKER", "ELECTRIC KETTLE", "TOASTER", "FAN", "IRON",
                  "COFFEE MAKER", "MICROWAVE", "SANDWICH MAKER", "VACUUM CLEANER"],
        "variants": ["1.5L VASO VIDRIO", "ANTIADHERENTE", "ACERO INOX", "PROGRAMABLE", "TURBO SILencio".upper(),
                     "CON PILA", "12 TAZAS", "DIGITAL", "RETRACTIL", "SIN BOLSA"],
        "sizes": ["220V", "1UN", "700W", "20L", "1.2L", "1200W"],
    },
    "HOME AND KITCHEN I": {
        "brands": ["CASA PRATICA", "COZINA DORADA", "MESA SERVIDA", "TERMOFLEx".upper(), "VAJILLASUR", "HOGARFINO"],
        "bases": ["DINNER PLATE SET", "DRINKING GLASSES", "COOKING POT", "FRYING PAN", "CUTLERY SET",
                  "FOOD CONTAINER", "MUG", "PITCHER", "BAKING TRAY", "KNIFE SET"],
        "variants": ["6 PIEZAS", "ANTIADHERENTE", "VIDRIO TEMPLADO", "ACERO", "TERMICO", "HERMETICO",
                     "DECORADO", "GRANDE", "PILSNER", "CON TAPA"],
        "sizes": ["6CT", "4CT", "24CM", "28CM", "1L", "12CT"],
    },
    "HOME AND KITCHEN II": {
        "brands": ["TEXTILO HOGAR", "SUENO BLANDO", "CONFORTAS", "LINO REAL", "ACOGEDOR", "TEJIDOS ANDINOS"],
        "bases": ["BED SHEET SET", "PILLOW", "BATH TOWEL", "BLANKET", "CURTAINS", "KITCHEN TOWEL SET",
                  "DOORMAT", "CUSHION COVER", "TABLECLOTH", "FACE TOWEL"],
        "variants": ["ALGODON", "MICROFIBRA", "IMPERMEABLE", "PLUSH", "ANTIDERRAMANTES", "FLEECE",
                     "BORDADO", "ANTIDESLIZANTE", "ESTAMPADO", "SUAVE TOUCH"],
        "sizes": ["1PLAZA", "2 PLAZAS", "70X140", "50X70", "6CT", "3CT"],
    },
    "HARDWARE": {
        "brands": ["HERRERIA PLUS", "TORNIFER", "BRIO MAX", "MECANIX", "ROBLETO", "FUERZA REAL"],
        "bases": ["SCREW SET", "HAMMER", "SCREWDRIVER SET", "DUCT TAPE", "PADLOCK", "NAILS",
                  "PLIERS", "DRILL BITS", "WRENCH", "PAINT BRUSH", "SANDPAPER"],
        "variants": ["FIBRA", "CABO DE MADERA", "AUTOPERFORANTE", "CROMO V", "COMBINADA", "SEGURIDAD",
                     "PUNTA MAGNETICA", "ALTA ADHERENCIA", "GRANO FINO", "AISLADA"],
        "sizes": ["20CT", "1UN", "6CT", "10M", "2CT", "50CT"],
    },
    "LAWN AND GARDEN": {
        "brands": ["JARDIN VIVO", "SEMILLASUR", "VERDECARPETO", "FLORAMAX", "RAICES", "HORTALIA"],
        "bases": ["GRASS SEED", "POTTING SOIL", "FLOWER POTS", "GARDEN HOSE", "PRUNING SHEARS",
                  "PLANT FERTILIZER", "WATERING CAN", "GARDEN GLOVES", "PLANT SEEDS", "PESTICIDE"],
        "variants": ["FERTILIZADO", "UNIVERSAL", "TERRACOTA", "FLEXIBLE 15M", "DE PODAR", "NPK",
                     "TROPICAL", "ANTIHIERBAS", "PARA INTERIOR", "ORGANICO"],
        "sizes": ["1KG", "5KG", "10L", "15M", "1UN", "500ML"],
    },
    "LINGERIE": {
        "brands": ["INTIMA", "SENSACION", "VELVETEA", "CORPELIA", "SUAVE NOCHE", "DIOSA"],
        "bases": ["BRA", "PANTIES 3PK", "PAJAMA SET", "SEAMLESS TOP", "SHAPEWEAR", "SOCKS",
                  "SPORTS BRA", "SLEEP SHIRT", "BOXER BRIEFS", "CAMISOLE"],
        "variants": ["ENCAJE", "ALGODON", "SATIN", "SIN COSTURAS", "DEPORTIVO", "MICROFIBRA",
                     "PUSH UP", "COMFORT", "ESTAMPADO", "LIVIANO"],
        "sizes": ["M", "S", "L", "UNICA", "XL", "9-10"],
    },
    "LADIESWEAR": {
        "brands": ["ELEGANZA", "MODA ANDINA", "VESTIR PLUS", "PRINCESA URBANA", "GLAM LINE", "SIENA"],
        "bases": ["BLOUSE", "DRESS", "JEANS", "CARDIGAN", "SKIRT", "LEGINGS".upper(),
                  "T-SHIRT", "HANDBAG", "SANDALS", "SCARF"],
        "variants": ["FLOREADO", "MEZCLILLA", "VERANO", "OFICINA", "CASUAL", "FIESTA",
                     "BORDADO", "BASICA", "CUERO SINTETICO", "ESTAMPADA"],
        "sizes": ["M", "S", "L", "UNICA", "38", "XL"],
    },
    "PLAYERS AND ELECTRONICS": {
        "brands": ["SONARIX", "TECNOVIA", "AUDIOPLUS", "PIXELMAX", "CONEXIA", "VOLTBEAT"],
        "bases": ["BLUETOOTH SPEAKER", "EARBUDS", "USB CABLE", "MEMORY CARD", "POWER BANK", "SMARTWATCH",
                  "PHONE CASE", "WALL CHARGER", "HDMI CABLE", "MP3 PLAYER"],
        "variants": ["PORTATIL", "INALAMBRICO", "TRENZADO", "CLASE 10", "10000MAH", "TACTIL",
                     "ANTIGOLPES", "RAPIDO 20W", "4K", "RADIO FM"],
        "sizes": ["1UN", "1M", "32GB", "2M", "64GB", "8GB"],
    },
    "CELEBRATION": {
        "brands": ["FIESTA ORO", "CELEBRA MAX", "CONFETI REAL", "ALEGRIA", "CUMPLEFELIZ", "BRINDIS"],
        "bases": ["BALLOON SET", "PARTY HATS", "GIFT WRAP", "CANDLES NUMBERS", "PINATA", "PARTY BAGS",
                  "BIRTHDAY BANNER", "SPARKLERS", "DISPOSABLE CUPS", "GIFT BAG"],
        "variants": ["SURTIDO", "DORADO", "INFANTIL", "CON MUSICA", "METALIZADO", "GRANDE",
                     "NUMEROS", "ARCOIRIS", "HELIO", " PERSONALIZADO".upper()],
        "sizes": ["12CT", "50CT", "1UN", "25CT", "6CT", "10CT"],
    },
}

# Títulos de respaldo si un family no está en el banco (no debería ocurrir con los 33)
DEFAULT_WORDS = {
    "brands": ["MARCA REAL", "SELECTO", "PRIMERA LINEA"],
    "bases": ["PRODUCTO"],
    "variants": ["ESPECIAL", "PREMIUM", "CLASICO"],
    "sizes": ["1UN", "STD"],
}


def make_name(rng: random.Random, family: str) -> str:
    """Ghép tên từ word bank theo pattern BRAND + BASE + VARIANT + SIZE."""
    words = FAMILY_WORDS.get(family, DEFAULT_WORDS)
    brand = rng.choice(words["brands"])
    base = rng.choice(words["bases"])
    name = f"{brand} {base}"
    # 80% có biến thể, 75% có size (độc lập) -> đa dạng tự nhiên
    if rng.random() < 0.80:
        name += f" {rng.choice(words['variants'])}"
    if rng.random() < 0.75:
        name += f" {rng.choice(words['sizes'])}"
    return name.replace("  ", " ").strip()


def generate_names() -> pd.DataFrame:
    items_df = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_nbr": "int64", "family": "str",
                                                         "class": "Int64", "perishable": "int64"})
    logger_rows = []
    used_names: set = set()
    names: list = []

    for item_nbr, family, class_code, perishable in items_df.itertuples(index=False, name=None):
        # Seed determinístico: cùng item_nbr -> cùng tên mọi lần chạy
        rng = random.Random(f"product-name:{item_nbr}:{class_code}")
        name = make_name(rng, family)
        # Chống trùng: ghém thêm biến thể khác, cuối cùng gắn mã class
        attempts = 0
        while name in used_names:
            words = FAMILY_WORDS.get(family, DEFAULT_WORDS)
            name = f"{make_name(rng, family)} {rng.choice(words['variants'])}".replace("  ", " ")
            attempts += 1
            if attempts > 5:
                name = f"{name} #{class_code}"
                break
        used_names.add(name)
        names.append(name)
        logger_rows.append((item_nbr, family, name))

    out = pd.DataFrame({"item_nbr": items_df["item_nbr"], "name": names})
    return out, logger_rows


def main() -> None:
    if not (RAW_DIR / "items.csv").exists():
        raise SystemExit(f"Không tìm thấy {RAW_DIR / 'items.csv'}")
    out, logger_rows = generate_names()

    assert out["name"].is_unique, "Tên sản phẩm trùng lặp!"
    assert out["name"].notna().all() and (out["name"].str.len() > 0).all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"[OK] Generated {len(out):,} unique product names -> {OUTPUT_PATH}")
    print("\n=== Mẫu mỗi family (tối đa 2 SP) ===")
    seen: dict = {}
    for item_nbr, family, name in logger_rows:
        seen.setdefault(family, []).append(name)
    for family in sorted(seen):
        for name in seen[family][:2]:
            print(f"  {family:<28} {name}")


if __name__ == "__main__":
    main()
