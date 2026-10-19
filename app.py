                st.rerun()
        if files and st.button("Process uploaded files", type="primary", use_container_width=True):
            new_records=[]
            for f in files:
                text=read_uploaded_file(f)
                if text and not text.startswith("[PDF extraction failed") and not text.startswith("[DOCX extraction failed"):
                    new_records.append(parse_resume(text,f.name))
            if new_records:
                st.session_state.resume_records.extend(new_records)
                st.success(f"Processed {len(new_records)} file(s).")
                st.rerun()
            else:
                st.error("No readable text extracted. Check the file format or use a text-based PDF.")
    with right:
        st.markdown("### Ranked candidate evidence")
        scored=[score_candidate(r,required,jd) for r in st.session_state.resume_records]
        scored=sorted(scored,key=lambda x:x["score"],reverse=True)
        if not scored:
            st.info("Add a resume to see candidate explanations.")
        for rank,cand in enumerate(scored,1):
            with st.container(border=True):
                top1,top2=st.columns([3,1])
                with top1:
                    st.markdown(f"**#{rank} · {cand['name']}**")
                    st.caption(f"{cand['years']:g} years detected · {len(cand['skills'])} normalized skills")
                with top2:
                    st.metric("Score",f"{cand['score']:.1f}/100")
                st.progress(min(max(cand["score"]/100,0),1))
                st.markdown("**Matched requirements**")
                if cand["matched"]:
                    st.markdown("".join(f'<span class="tag">{s}</span>' for s in cand["matched"]),unsafe_allow_html=True)
                else: st.caption("No required skills matched in extracted text.")
                st.markdown("**Missing requirements**")
                st.write(", ".join(cand["missing"]) if cand["missing"] else "No missing skills detected.")
                with st.expander("Why this score? · Evidence"):
                    x,y=st.columns(2)
                    x.metric("Skill coverage contribution",f"{cand['skill_points']:.1f}",f"{len(cand['matched'])}/{len(required)} required skills")
                    y.metric("Text similarity contribution",f"{cand['text_points']:.1f}", "TF-IDF cosine similarity")
                    st.caption("These are model components, not causal explanations or proof of candidate ability.")
                    st.markdown("**Anonymized resume preview**")
                    st.code(cand["redacted"][:2500] or "No text extracted")
                    st.markdown("**Extracted skills**")
                    st.write(", ".join(cand["skills"]) or "None detected")
        if scored:
            export=pd.DataFrame([{"rank":i+1,"candidate":x["name"],"score":x["score"],"matched_skills":", ".join(x["matched"]),"missing_skills":", ".join(x["missing"])} for i,x in enumerate(scored)])
            st.download_button("Export review CSV",export.to_csv(index=False).encode(),file_name="fairhire_review.csv",mime="text/csv")

elif page == "Fairness audit":
    st.subheader("Fairness Audit Lab")
    st.caption("A controlled simulation to demonstrate group-level metrics. These synthetic groups are not real demographic data and do not establish real-world fairness.")
    with st.container(border=True):
        st.markdown("### Configure a stress test")
        c1,c2,c3=st.columns(3)
        with c1: n=st.slider("Candidates per group",20,500,120,step=20)
        with c2: threshold=st.slider("Selection threshold",30,90,65)
        with c3: bias=st.slider("Injected score offset for Group B",0,30,0,help="An intentionally injected score offset used only to verify that the audit detects a disparity.")
        run=st.button("Run synthetic audit",type="primary")
    if "audit_df" not in st.session_state or run:
        st.session_state.audit_df=synthetic_audit(n,threshold,bias)
        st.session_state.audit_config={"n":n,"threshold":threshold,"bias":bias}
    df=st.session_state.audit_df
    fm=fairness_metrics(df)
    c1,c2,c3,c4=st.columns(4)
    dp=fm["dp_difference"]; di=fm["di_ratio"]
    c1.metric("Demographic parity difference",f"{dp:.3f}" if pd.notna(dp) else "N/A",help="Maximum minus minimum group selection rate; closer to 0 means rates are more similar.")
    c2.metric("Disparate impact ratio",f"{di:.3f}" if pd.notna(di) else "N/A",help="Minimum group selection rate divided by maximum group selection rate.")
    c3.metric("Selection threshold",str(st.session_state.audit_config["threshold"]))
    c4.metric("Synthetic applicants",str(len(df)))
    l,r=st.columns([1,1])
    with l:
        rates=pd.DataFrame([{"Group":g,"Selection rate":rate} for g,rate in fm["rates"].items()])
        fig=px.bar(rates,x="Group",y="Selection rate",text=rates["Selection rate"].map(lambda v:f"{v:.1%}"),range_y=[0,1],color="Group")
        fig.update_layout(height=330,showlegend=False,plot_bgcolor="white",paper_bgcolor="white",margin=dict(l=5,r=5,t=25,b=5))
        st.plotly_chart(fig,use_container_width=True)
    with r:
        fig2=px.histogram(df,x="qualification_score",color="group",nbins=20,barmode="overlay",opacity=.7,
                          title="Synthetic qualification score distribution")
        fig2.update_layout(height=330,plot_bgcolor="white",paper_bgcolor="white",margin=dict(l=5,r=5,t=50,b=5))
        st.plotly_chart(fig2,use_container_width=True)
    st.markdown("### Group-level results")
    summary=df.groupby("group").agg(candidates=("candidate_id","count"),selected=("selected","sum"),selection_rate=("selected","mean"),mean_score=("qualification_score","mean")).reset_index()
    summary["selection_rate"]=summary["selection_rate"].map(lambda x:f"{x:.1%}")
    summary["mean_score"]=summary["mean_score"].map(lambda x:f"{x:.1f}")
    st.dataframe(summary,use_container_width=True,hide_index=True)
    if pd.isna(di):
        st.info("No group has a positive selection rate, so the ratio is undefined.")
    elif di >= .8 and dp < .1:
        st.success("This run shows relatively similar group selection rates under the configured synthetic scenario. This is not proof of fairness.")
    else:
        st.warning("This run flags a selection-rate disparity worth investigating. Check the score distributions, threshold, sample size, and any injected offset.")
    with st.expander("Metric definitions and caveats"):
        st.markdown("""
        - **Demographic parity difference:** largest group selection rate minus smallest group selection rate. Zero means equal selection rates in this simulation.
        - **Disparate impact ratio:** smallest group selection rate divided by the largest. A value of 1 means equal selection rates; the often-cited 0.80 heuristic is not a universal legal test.
        - These metrics can conceal important differences within groups and do not establish that a hiring process is valid, fair, or lawful.
        - The simulation uses a controlled score distribution and optional artificial offset. It is a unit-test-style demonstration, not an empirical audit of an employer.
        """)
    st.download_button("Download audit data (CSV)",df.to_csv(index=False).encode(),file_name="fairhire_synthetic_audit.csv",mime="text/csv")

else:
    st.subheader("Methodology & responsible use")
    st.caption("Transparent prototype documentation")
    st.markdown("### Scoring model")
    st.latex(r"S = 75\\left(\\frac{|M \\cap R|}{|R|}\\right) + 25\\,\\operatorname{cosine}(v_{JD}, v_{resume})")
    st.markdown("""
    - **Skill coverage (75 points):** proportion of configured required skills detected by a transparent alias dictionary.
    - **Text similarity (25 points):** TF-IDF cosine similarity between job description and resume text.
    - **Evidence:** matched and missing required skills are shown for every candidate.
    - **Privacy:** common contact patterns and demo names are masked in the preview. This is not a complete PII detector.
    - **Fairness:** synthetic paired/stress-test data help verify metrics; real-world auditing requires lawful, carefully governed data and appropriate expertise.
    """)
    st.markdown("### Known limitations")
    limitations=[
        ("Skill extraction is dictionary-based","Synonyms, context, proficiency, and false matches can be missed."),
        ("Text similarity is not job competence","A similar resume may score well without meeting essential requirements."),
        ("PII redaction is best-effort","The prototype does not guarantee removal of all names, locations, dates, or indirect identifiers."),
        ("Fairness metrics are incomplete","Aggregate parity does not capture every form of discrimination or intersectional effect."),
        ("No automated hiring decisions","Use results as review assistance only; keep a qualified human decision-maker involved."),
    ]
    for title,body in limitations:
        with st.expander(title): st.write(body)
    st.markdown("### Recommended next technical upgrades")
    st.markdown("1. Add evaluation fixtures and tests for skill extraction, redaction, and counterfactual invariance.")
    st.markdown("2. Add calibrated semantic embeddings with a documented local model and reproducible versioning.")
    st.markdown("3. Add a stronger PII recognizer and redaction recall/precision evaluation.")
    st.markdown("4. Add model cards, audit logs, role-based access, retention controls, and secure deployment.")
    st.markdown("5. Validate against representative, legally obtained data before any real-world use.")
