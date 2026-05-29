#!/usr/bin/env python3
"""
TrovaMUZ Caption Enrichment Tool — v2.0
========================================
Gen Audius LLC · musicgau_adn

Pipeline por archivo:
  1. librosa        → BPM, key, modo, energía (fuente PRIMARIA)
  2. Shazam         → identificar artista / título / año
  3. Spotify        → solo metadata (géneros, popularidad) — audio features deprecados Nov 2024
  4. Last.fm        → tags curados por humanos
  5. GPT-4o         → sintetiza todo en caption final ultra-rico
  6. gpt-image-2    → genera portada artística por género

Uso:
  python tools/enrich_captions.py --folder training/datasets/bachata_moderna
  python tools/enrich_captions.py --all --dry-run
  python tools/enrich_captions.py --all --overwrite
  python tools/enrich_captions.py --all --covers
  python tools/enrich_captions.py --all --no-gpt   # solo reglas, sin GPT
"""

import os, sys, json, time, base64, argparse, textwrap
from pathlib import Path

import numpy as np
import requests

try:
    import librosa as lr
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("⚠  librosa no instalado — pip install librosa")

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_OK = True
except ImportError:
    SPOTIPY_OK = False

try:
    import pylast
    PYLAST_OK = True
except ImportError:
    PYLAST_OK = False

try:
    from openai import OpenAI as _OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Credenciales ───────────────────────────────────────────────────────────────
RAPIDAPI_KEY          = os.environ.get("RAPIDAPI_KEY", "")
SPOTIFY_CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
LASTFM_API_KEY        = os.environ.get("LASTFM_API_KEY", "")
LASTFM_API_SECRET     = os.environ.get("LASTFM_API_SECRET", "")
OPENAI_API_KEY        = os.environ.get("OPENAI_API_KEY", "")

AUDIO_EXTS   = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MUSICAL_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ── TRIGGER WORD — siempre al inicio de cada caption ─────────────────────────
TRIGGER = "musicgau_adn style, TrovaMUZ_V1"

# ── Captions base por género (descripción técnica de instrumentos) ─────────────
GENRE_BASE = {
    "bolero_bachata": (
        "bolero bachata romántica, warm nylon-string requinto guitar lead, "
        "rhythm guitar strumming, slap bass on beat 4, bongo open tones, "
        "guira sixteenth notes, intimate male vocals"
    ),
    "bachata_moderna": (
        "modern bachata, electric requinto guitar lead with slides, "
        "rhythm guitar chord strumming, slap bass syncopated groove, "
        "bongo martillo pattern, guira texture, contemporary production"
    ),
    "bachata_tradicional": (
        "traditional Dominican bachata, acoustic requinto guitar, "
        "double bass pattern, maracas, bongo, bolero influence, "
        "classic romantic vocals"
    ),
    "bachata_amargue": (
        "bachata amargue, bitter romantic style, raw emotional requinto guitar, "
        "traditional Dominican rhythm, heartbreak vocals, classic acoustic"
    ),
    "merengue_tipico": (
        "merengue tipico perico ripiao, accordion-driven, Dominican folk, "
        "guira metal scraper, tambora drum, countryside traditional style"
    ),
    "merengue_80s": (
        "merengue 1980s vintage, bright accordion melody, brass section, "
        "Dominican dance, upbeat classic arrangement, festive"
    ),
    "merengue_90s": (
        "merengue 1990s commercial, synthesizer leads, modern production, "
        "danceable festive, Dominican pop dance"
    ),
    "merengue": (
        "merengue, Dominican rhythm, accordion lead, guira scraper, "
        "tambora drum, brass stabs, festive upbeat dance"
    ),
    "salsa": (
        "salsa, piano montuno on clave pattern, bass tumbao, conga percussion, "
        "trumpet and trombone brass section, coro vocals, Latin jazz influence"
    ),
    "salsa_dura": (
        "salsa dura New York style, aggressive piano montuno, driving bass tumbao, "
        "powerful brass stabs, strong conga, call and response coro"
    ),
    "salsa_romantica": (
        "salsa romantica, smooth piano, elegant strings, subtle brass, "
        "warm conga, intimate vocals, romantic mood"
    ),
    "salsa_corta_vena": (
        "salsa corta vena, heavy powerful brass, intense clave, "
        "soulful piano montuno, street salsa, Puerto Rican influence"
    ),
    "cumbia": (
        "Colombian cumbia, accordion lead melody, caja drum downbeat, "
        "guacharaca scraper, bass guitar, maracas, tropical folk"
    ),
    "vallenato": (
        "Colombian vallenato, diatonic accordion ornamental runs, "
        "caja drum traditional pattern, guacharaca texture, "
        "storytelling vocals, Sierra Nevada style"
    ),
    "reggaeton": (
        "reggaeton urbano, dembow rhythm, 808 sub bass, hi-hat triplets, "
        "synth brass stabs, auto-tune vocals, Latin urban trap"
    ),
    "pop_latino": (
        "Latin pop, acoustic guitar strumming, electronic drums, synth bass, "
        "warm pad chords, catchy vocals, commercial production"
    ),
    "bolero_del_ayer": (
        "bolero clasico, vintage romantic Latin ballad, lush orchestral strings, "
        "maracas, bongo, classic crooner vocal style, 1950s golden era"
    ),
    # ── Alias para nombres de carpeta reales en E:\TrovaMUZ_V1\dataset ───────────
    "bachata": (
        "Dominican bachata, syncopated bicheo rhythm, requinto guitar lead with expressive bends, "
        "rhythmic segunda guitar, slap bass on beat 4, bongo martillo, bright guira scraper, "
        "intimate studio sound"
    ),
    "bachata-de-amargue": (
        "bachata de amargue, bitter melancholic Dominican bachata, raw emotional requinto bends, "
        "punching bongo martillo pattern, bright guira, dark heartbreak vocals, "
        "classic acoustic vintage studio"
    ),
    "boleros": (
        "bolero romántico latinoamericano, warm nylon-string guitar, piano accompaniment, "
        "light maracas, lush romantic atmosphere, ballad tempo, expressive emotional vocals"
    ),
    "cumbia": (
        "Colombian cumbia, accordion lead melody, caja drum downbeat, "
        "guacharaca scraper, bass guitar, maracas, tropical folk dance"
    ),
    "merengue clasico": (
        "merengue clasico dominicano, bright accordion melody, brass section stabs, "
        "tambora drum two-beat pulse, metal guira scraper, festive upbeat Caribbean dance"
    ),
    "pop baladas contemporaneas": (
        "Latin pop balada contemporanea, acoustic and electric guitar layers, "
        "electronic drums, warm synth pads, commercial studio mix, emotional catchy vocals"
    ),
    "reguetones": (
        "reggaeton urbano latino, dembow rhythm pattern, 808 sub bass, hi-hat triplets, "
        "synth brass stabs, trap influences, auto-tune vocals, modern Latin urban production"
    ),
    "salsa": (
        "salsa Latina, piano montuno on clave pattern, bass tumbao, conga percussion, "
        "trumpet and trombone brass section, coro call-and-response vocals, Latin jazz influence"
    ),
    "vallenatos": (
        "Colombian vallenato, diatonic button accordion ornamental runs, "
        "caja drum traditional pattern, guacharaca texture, storytelling vocals, "
        "Sierra Nevada folk style"
    ),
}

def get_genre_base(genre_hint: str) -> str:
    key = genre_hint.lower().strip()
    if key in GENRE_BASE:
        return GENRE_BASE[key]
    for k, v in GENRE_BASE.items():
        if k in key or key in k:
            return v
    return genre_hint.replace("_", " ")


# ── librosa — FUENTE PRIMARIA de BPM, key y energía ──────────────────────────

def analyze_locally(wav_path: Path) -> dict:
    """Extrae BPM, tonalidad, modo y energía directamente del audio."""
    if not LIBROSA_OK:
        return {}
    try:
        # Duración total
        total  = lr.get_duration(path=str(wav_path))
        offset = min(total * 0.15, 20.0)
        load_dur = min(90.0, total * 0.7)

        audio, sr = lr.load(str(wav_path), sr=None, mono=True,
                            offset=offset, duration=load_dur)

        # BPM — beat_track sin units="time" para obtener BPM directamente
        tempo, _ = lr.beat.beat_track(y=audio, sr=sr)
        bpm = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
        # Corregir errores de octava comunes
        while bpm > 180: bpm /= 2.0
        while bpm < 60:  bpm *= 2.0

        # Key + Mode — Krumhansl-Schmuckler profiles
        chroma   = lr.feature.chroma_cqt(y=audio, sr=sr)
        chroma_m = np.mean(chroma, axis=1)
        major_p  = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_p  = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        maj_scores = [np.corrcoef(np.roll(chroma_m, -i), major_p)[0, 1] for i in range(12)]
        min_scores = [np.corrcoef(np.roll(chroma_m, -i), minor_p)[0, 1] for i in range(12)]
        if max(maj_scores) >= max(min_scores):
            mode    = "major"
            key_idx = int(np.argmax(maj_scores))
        else:
            mode    = "minor"
            key_idx = int(np.argmax(min_scores))

        # Energía + LUFS aproximado
        rms    = float(np.mean(lr.feature.rms(y=audio)))
        energy = min(rms / 0.15, 1.0)
        lufs   = round(20 * np.log10(max(rms, 1e-10)), 1)

        return {
            "bpm":    round(bpm),
            "key":    f"{MUSICAL_KEYS[key_idx]} {mode}",
            "energy": energy,
            "lufs":   lufs,
        }
    except Exception as e:
        print(f"    [librosa] Error: {e}")
        return {}


# ── Shazam — endpoint 2025 ─────────────────────────────────────────────────────

def identify_with_shazam(wav_path: Path) -> dict | None:
    if not RAPIDAPI_KEY or not LIBROSA_OK:
        return None
    try:
        total  = lr.get_duration(path=str(wav_path))
        offset = min(total * 0.3, 30.0)
        audio, _ = lr.load(str(wav_path), sr=44100, mono=True,
                           offset=offset, duration=10.0)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        b64 = base64.b64encode(pcm.tobytes()).decode("utf-8")

        resp = requests.post(
            "https://shazam.p.rapidapi.com/songs/recognize",
            data=b64,
            headers={
                "content-type":    "text/plain",
                "X-RapidAPI-Key":  RAPIDAPI_KEY,
                "X-RapidAPI-Host": "shazam.p.rapidapi.com",
            },
            timeout=20,
        )
        if resp.status_code == 200:
            return resp.json().get("track")
    except Exception as e:
        print(f"    [Shazam] Error: {e}")
    return None


# ── Spotify — solo metadata (audio features deprecados Nov 2024) ───────────────

def _get_spotify_client():
    if not SPOTIPY_OK or not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    try:
        return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
        ))
    except Exception:
        return None


def get_spotify_metadata(artist: str, title: str, sp) -> dict | None:
    if sp is None or not artist:
        return None
    try:
        results = sp.search(q=f"artist:{artist} track:{title}", type="track", limit=1)
        items   = results.get("tracks", {}).get("items", [])
        if not items:
            results = sp.search(q=f"{title} {artist}", type="track", limit=1)
            items   = results.get("tracks", {}).get("items", [])
        if not items:
            return None
        track  = items[0]
        art_id = track["artists"][0]["id"] if track.get("artists") else None
        genres = []
        if art_id:
            genres = sp.artist(art_id).get("genres", [])[:3]
        return {
            "popularity": track.get("popularity", 0),
            "year":       (track.get("album", {}).get("release_date", "") or "")[:4],
            "genres":     genres,
        }
    except Exception as e:
        print(f"    [Spotify] Error: {e}")
        return None


# ── Last.fm ────────────────────────────────────────────────────────────────────

NOISE_TAGS = {
    "seen live", "albums i own", "favourites", "favorite", "love",
    "awesome", "good", "music", "spotify", "youtube", "amazing",
    "cool", "great", "best", "beautiful",
}

def _get_lastfm_network():
    if not PYLAST_OK or not LASTFM_API_KEY:
        return None
    try:
        return pylast.LastFMNetwork(api_key=LASTFM_API_KEY, api_secret=LASTFM_API_SECRET or "")
    except Exception:
        return None


def get_lastfm_tags(artist: str, title: str, network) -> list[str]:
    if network is None:
        return []
    tags = []
    try:
        top  = network.get_track(artist, title).get_top_tags(limit=15)
        tags = [t.item.name.lower() for t in top]
    except Exception:
        pass
    if not tags and artist:
        try:
            top  = network.get_artist(artist).get_top_tags(limit=15)
            tags = [t.item.name.lower() for t in top]
        except Exception:
            pass
    return [t for t in tags if t not in NOISE_TAGS and len(t) > 2][:8]


def get_lastfm_genre_tags(genre_hint: str, network) -> list[str]:
    if network is None:
        return []
    try:
        sims = network.get_tag(genre_hint.replace("_", " ")).get_similar()
        return [s.item.name.lower() for s in sims[:6]]
    except Exception:
        return []


# ── Caption builder (reglas) ──────────────────────────────────────────────────

MOOD_TAGS = {
    "romantic", "melancholic", "sad", "happy", "energetic", "festive",
    "intimate", "powerful", "dark", "nostalgic", "sensual", "emotional",
    "upbeat", "mellow", "soulful", "passionate", "tender", "joyful",
    "heartbreak", "longing",
}

def build_caption_rules(
    genre_hint:   str,
    shazam_track: dict | None,
    spotify_meta: dict | None,
    lastfm_tags:  list[str],
    local:        dict,
) -> str:
    """Caption basado en reglas — se usa como fallback o base para GPT."""
    parts = [TRIGGER, get_genre_base(genre_hint)]

    # Mood de Last.fm
    mood = [t for t in lastfm_tags if t in MOOD_TAGS][:2]
    if mood:
        parts.append(", ".join(mood))

    # BPM y key de librosa
    if local.get("bpm"):
        parts.append(f"{local['bpm']} BPM")
    if local.get("key"):
        parts.append(local["key"])

    # Energía
    e = local.get("energy")
    if e is not None:
        parts.append("high energy" if e > 0.75 else ("medium energy" if e > 0.45 else "soft intimate"))

    # Tags extra de Last.fm
    extra = [t for t in lastfm_tags if t not in MOOD_TAGS][:3]
    if extra:
        parts.append(", ".join(extra))

    # Año de Shazam
    if shazam_track:
        for sec in shazam_track.get("sections", []):
            for meta in sec.get("metadata", []):
                if meta.get("title") == "Released":
                    yr = (meta.get("text", "") or "")[:4]
                    if yr.isdigit() and 1950 <= int(yr) <= 2030:
                        parts.append(yr)

    parts.append("high fidelity audio, professional studio quality")
    return ", ".join(parts)


# ── OpenAI — GPT-4o caption + gpt-image-2 portadas ───────────────────────────

def _get_openai_client():
    if not OPENAI_OK or not OPENAI_API_KEY:
        return None
    return _OpenAI(api_key=OPENAI_API_KEY)


def enrich_caption_with_gpt(
    genre_hint:   str,
    base_caption: str,
    spotify_meta: dict | None,
    lastfm_tags:  list[str],
    local:        dict,
    shazam_track: dict | None,
    client,
) -> str:
    if client is None:
        return base_caption
    meta = {
        "genre":            genre_hint.replace("_", " "),
        "trigger_word":     TRIGGER,
        "base_description": base_caption,
        "bpm":              local.get("bpm"),
        "key":              local.get("key"),
        "energy":           round(local.get("energy", 0), 2),
        "lastfm_tags":      lastfm_tags[:8],
    }
    if spotify_meta:
        meta["spotify"] = spotify_meta
    if shazam_track:
        meta["identified_as"] = f"{shazam_track.get('subtitle','')} - {shazam_track.get('title','')}"

    system = (
        "You are an expert in Latin music production and AI music model training. "
        f"Always start the caption with exactly '{TRIGGER} ' (followed by a SPACE, never a comma). "
        "Then continue with a comma-separated MusicGen Melody training caption that describes: "
        "specific instruments and their roles, rhythm feel, harmonic key, energy, "
        "mood, and production quality. Be specific and musical. "
        "No explanations, no quotes, no headers. Maximum 70 words total. "
        f"Example format: '{TRIGGER} bachata romántica, requinto guitar lead, 118 BPM, D minor, intimate studio'"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Metadata:\n{json.dumps(meta, ensure_ascii=False, indent=2)}"},
            ],
            max_tokens=200,
            temperature=0.35,
        )
        result = resp.choices[0].message.content.strip().strip('"').strip("'")
        # Ensure no comma immediately after trigger word
        import re as _re
        result = _re.sub(r"(TrovaMUZ_V1),\s*", r"\1 ", result)
        return result
    except Exception as e:
        print(f"    [GPT-4o] Error: {e}")
        return base_caption


def generate_cover_art(genre_hint: str, caption: str, output_path: Path, client) -> bool:
    if client is None:
        return False
    dalle_prompt = (
        f"Professional music album cover art for {genre_hint.replace('_',' ')} Latin music. "
        f"Style: {caption[:100]}. "
        "Vibrant tropical Caribbean colors, artistic illustration, "
        "no text, no letters, no words anywhere, high quality digital art, square."
    )
    for model in ["gpt-image-2", "gpt-image-1", "chatgpt-image-latest"]:
        try:
            resp = client.images.generate(
                model=model, prompt=dalle_prompt, size="1024x1024", n=1
            )
            item = resp.data[0]
            if getattr(item, "url", None):
                img = requests.get(item.url, timeout=30).content
            elif getattr(item, "b64_json", None):
                img = base64.b64decode(item.b64_json)
            else:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(img)
            print(f"    [{model}] Portada guardada → {output_path.name} ({len(img)//1024}KB)")
            return True
        except Exception as e:
            print(f"    [{model}] No disponible: {str(e)[:60]}")
    return False


# ── Procesamiento por carpeta ─────────────────────────────────────────────────

def process_folder(
    folder:          Path,
    dry_run:         bool,
    sp,
    lastfm_net,
    overwrite:       bool,
    openai_client    = None,
    generate_covers: bool = False,
):
    audio_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        print(f"  Sin archivos de audio en {folder.name}")
        return

    genre_hint = folder.name
    print(f"\n{'='*65}")
    print(f"  Género : {genre_hint}  ({len(audio_files)} archivos)")
    print(f"{'='*65}")

    last_caption = ""
    ok = 0

    for wav_path in audio_files:
        txt_path = wav_path.with_suffix(".txt")

        if txt_path.exists() and not overwrite:
            print(f"  [SKIP] {wav_path.name}")
            continue

        print(f"\n  → {wav_path.name}")

        # 1. librosa (primario)
        local = analyze_locally(wav_path)
        if local:
            print(f"    librosa : BPM={local.get('bpm')} | Key={local.get('key')} | LUFS={local.get('lufs')}")

        # 2. Shazam
        shazam_track, artist, title = None, "", ""
        if RAPIDAPI_KEY:
            shazam_track = identify_with_shazam(wav_path)
            if shazam_track:
                artist = shazam_track.get("subtitle", "")
                title  = shazam_track.get("title", "")
                print(f"    Shazam  : ✅ {artist} — {title}")
            else:
                print("    Shazam  : no identificado")
            time.sleep(0.6)

        # 3. Spotify metadata (no audio features)
        spotify_meta = None
        if artist and title:
            spotify_meta = get_spotify_metadata(artist, title, sp)
            if spotify_meta:
                print(f"    Spotify : year={spotify_meta.get('year')} genres={spotify_meta.get('genres')}")
            time.sleep(0.3)

        # 4. Last.fm tags
        lastfm_tags = []
        if artist and title:
            lastfm_tags = get_lastfm_tags(artist, title, lastfm_net)
        if not lastfm_tags:
            lastfm_tags = get_lastfm_genre_tags(genre_hint, lastfm_net)
        if lastfm_tags:
            print(f"    Last.fm : {', '.join(lastfm_tags[:5])}")
        time.sleep(0.3)

        # 5. Caption base (reglas)
        base_caption = build_caption_rules(genre_hint, shazam_track, spotify_meta, lastfm_tags, local)

        # 6. Enriquecer con GPT-4o
        if openai_client:
            print("    GPT-4o  : enriqueciendo...")
            caption = enrich_caption_with_gpt(
                genre_hint, base_caption, spotify_meta,
                lastfm_tags, local, shazam_track, openai_client
            )
            print(f"    Caption : {caption[:90]}...")
        else:
            caption = base_caption
            print(f"    Caption : {caption[:90]}...")

        last_caption = caption

        if not dry_run:
            txt_path.write_text(caption, encoding="utf-8")
            print(f"    ✅ {txt_path.name}")
            ok += 1
        else:
            print(f"    [DRY RUN] → {txt_path.name}")
            ok += 1

    # 7. Portada DALL-E (una por género)
    if generate_covers and openai_client:
        cover_path = Path(__file__).resolve().parent.parent / "covers" / f"{genre_hint}.png"
        if not cover_path.exists() or overwrite:
            print(f"\n  Generando portada para {genre_hint}...")
            if not dry_run:
                generate_cover_art(genre_hint, last_caption, cover_path, openai_client)
                if cover_path.exists():
                    import shutil
                    shutil.copy2(cover_path, folder / "cover.png")
            else:
                print(f"  [DRY RUN] generaría portada → {cover_path.name}")
        else:
            print(f"  [SKIP] Portada existe: {cover_path.name}")

    print(f"\n  Procesados: {ok}/{len(audio_files)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TrovaMUZ v2.0 — Enriquecimiento automático de captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Ejemplos:
              python tools/enrich_captions.py --folder training/datasets/bachata_amargue
              python tools/enrich_captions.py --all --covers
              python tools/enrich_captions.py --all --dry-run
              python tools/enrich_captions.py --all --overwrite --covers
        """),
    )
    parser.add_argument("--folder",    type=str,        help="Carpeta de un género específico")
    parser.add_argument("--all",       action="store_true", help="Procesar todas las carpetas en training/datasets/")
    parser.add_argument("--dry-run",   action="store_true", help="Vista previa sin escribir archivos")
    parser.add_argument("--overwrite", action="store_true", help="Reemplazar captions existentes")
    parser.add_argument("--covers",    action="store_true", help="Generar portada por género con gpt-image-2")
    parser.add_argument("--no-gpt",   action="store_true", help="Omitir GPT-4o, usar solo reglas")
    args = parser.parse_args()

    if not args.folder and not args.all:
        parser.print_help()
        sys.exit(1)

    print("\n🎵 TrovaMUZ Caption Enrichment v2.0 — Gen Audius LLC")
    print(f"   Trigger : '{TRIGGER}'\n")

    sp         = _get_spotify_client()
    lastfm_net = _get_lastfm_network()
    openai_cli = None if args.no_gpt else _get_openai_client()

    print(f"  {'✅' if LIBROSA_OK   else '❌'} librosa   (BPM + key primario)")
    print(f"  {'✅' if RAPIDAPI_KEY else '⚠ '} Shazam    (identificación)")
    print(f"  {'✅' if sp           else '⚠ '} Spotify   (metadata)")
    print(f"  {'✅' if lastfm_net   else '⚠ '} Last.fm   (tags)")
    print(f"  {'✅' if openai_cli   else '⚠ '} GPT-4o    (caption enriquecido)")
    print(f"  {'✅' if (openai_cli and args.covers) else '⚠ '} gpt-image-2 (portadas)\n")

    root = Path(__file__).resolve().parent.parent / "training" / "datasets"

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_absolute():
            folder = (Path(__file__).resolve().parent.parent / args.folder).resolve()
        if not folder.exists():
            print(f"ERROR: carpeta no encontrada: {folder}")
            sys.exit(1)
        process_folder(folder, args.dry_run, sp, lastfm_net, args.overwrite,
                       openai_cli, args.covers)
    else:
        folders = sorted(
            p for p in root.iterdir()
            if p.is_dir() and p.name != "combined"
        )
        if not folders:
            print(f"No se encontraron subcarpetas en {root}")
            sys.exit(1)
        for folder in folders:
            process_folder(folder, args.dry_run, sp, lastfm_net, args.overwrite,
                           openai_cli, args.covers)

    print("\n✅ Listo.")


if __name__ == "__main__":
    main()
