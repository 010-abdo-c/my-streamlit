import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# دالة FCFS
def fcfs(processes):
    processes = sorted(processes, key=lambda x: x['arrival'])
    t = 0
    timeline = []
    wait_times = []
    
    for p in processes:
        if t < p['arrival']:
            t = p['arrival']
        start = t
        wait = start - p['arrival']
        finish = t + p['burst']
        t = finish
        timeline.append({'pid': p['pid'], 'start': start, 'finish': finish})
        wait_times.append(wait)
    
    return timeline, sum(wait_times) / len(wait_times)

# دالة SJF
def sjf(processes):
    processes = sorted(processes, key=lambda x: x['arrival'])
    t = 0
    remain = processes.copy()
    timeline = []
    wait_times = []

    while remain:
        ready = [p for p in remain if p['arrival'] <= t]
        if not ready:
            t = min(remain, key=lambda x: x['arrival'])['arrival']
            continue

        p = min(ready, key=lambda x: x['burst'])
        start = t
        wait = start - p['arrival']
        finish = t + p['burst']
        t = finish

        timeline.append({'pid': p['pid'], 'start': start, 'finish': finish})
        wait_times.append(wait)
        remain.remove(p)

    return timeline, sum(wait_times) / len(wait_times)

# دالة Round Robin
def round_robin(processes, quantum):
    from collections import deque
    t = 0
    queue = deque()
    remain = [{'pid': p['pid'], 'arrival': p['arrival'], 'burst': p['burst'], 'left': p['burst']} for p in processes]
    timeline = []
    wait_times = {p['pid']: 0 for p in processes}
    last_time = {p['pid']: p['arrival'] for p in processes}

    while remain or queue:
        for p in list(remain):
            if p['arrival'] <= t:
                queue.append(p)
                remain.remove(p)

        if not queue:
            t += 1
            continue

        p = queue.popleft()
        start = t
        
        wait_times[p['pid']] += start - last_time[p['pid']]
        
        run = min(quantum, p['left'])
        t += run
        p['left'] -= run
        finish = t

        timeline.append({'pid': p['pid'], 'start': start, 'finish': finish})
        last_time[p['pid']] = finish

        for x in list(remain):
            if x['arrival'] <= t:
                queue.append(x)
                remain.remove(x)

        if p['left'] > 0:
            queue.append(p)

    avg_wait = sum(wait_times.values()) / len(wait_times)
    return timeline, avg_wait

# رسم Gantt Chart باستخدام Plotly
def draw_gantt(timeline, title):
    df = pd.DataFrame(timeline)
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['finish'] - row['start']],
            y=['CPU'],
            orientation='h',
            base=row['start'],
            marker=dict(color=colors[(row['pid']-1) % len(colors)]),
            text=f"P{row['pid']}",
            textposition='inside',
            name=f"Process {row['pid']}",
            showlegend=False,
            hovertemplate=f"<b>Process {row['pid']}</b><br>Start: {row['start']}<br>Finish: {row['finish']}<br>Duration: {row['finish']-row['start']}<extra></extra>"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="الوقت (Time)",
        yaxis_title="",
        height=200,
        barmode='stack',
        showlegend=False,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# واجهة التطبيق
st.set_page_config(page_title="CPU Scheduler", layout="wide", page_icon="🖥️")

st.title("🖥️ محاكي جدولة المعالج - CPU Scheduling Simulator")
st.markdown("---")

# إنشاء جدول البيانات الافتراضي
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        'Process': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'Arrival Time': [0, 1, 2, 3, 4],
        'Burst Time': [5, 3, 8, 6, 4],
        'Priority': [2, 1, 3, 2, 4]
    })

st.subheader("📊 جدول العمليات - Process Table")
st.markdown("*قم بتعديل القيم في الجدول مباشرة*")

# عرض الجدول القابل للتعديل
edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Process": st.column_config.TextColumn("العملية", disabled=True),
        "Arrival Time": st.column_config.NumberColumn("وقت الوصول", min_value=0, max_value=100, step=1),
        "Burst Time": st.column_config.NumberColumn("وقت التنفيذ", min_value=1, max_value=100, step=1),
        "Priority": st.column_config.NumberColumn("الأولوية", min_value=1, max_value=10, step=1)
    }
)

st.session_state.df = edited_df

# إعداد Round Robin Quantum
col1, col2 = st.columns([3, 1])
with col2:
    quantum = st.number_input("⏱️ Round Robin Quantum", min_value=1, value=2, step=1)

st.markdown("---")

# تحويل البيانات للمعالجة
processes = []
for idx, row in edited_df.iterrows():
    processes.append({
        'pid': idx + 1,
        'arrival': int(row['Arrival Time']),
        'burst': int(row['Burst Time']),
        'priority': int(row['Priority'])
    })

# حساب النتائج
fcfs_timeline, fcfs_wait = fcfs(processes)
sjf_timeline, sjf_wait = sjf(processes)
rr_timeline, rr_wait = round_robin(processes, quantum)

# عرض النتائج في تبويبات
tab1, tab2, tab3, tab4 = st.tabs(["📈 FCFS", "📊 SJF", "🔄 Round Robin", "📉 المقارنة"])

with tab1:
    st.subheader("First Come First Serve (FCFS)")
    draw_gantt(fcfs_timeline, "FCFS Gantt Chart")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("متوسط وقت الانتظار", f"{fcfs_wait:.2f}")
    with col2:
        st.metric("عدد العمليات", len(processes))
    with col3:
        total_time = max([t['finish'] for t in fcfs_timeline])
        st.metric("إجمالي الوقت", total_time)

with tab2:
    st.subheader("Shortest Job First (SJF)")
    draw_gantt(sjf_timeline, "SJF Gantt Chart")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("متوسط وقت الانتظار", f"{sjf_wait:.2f}")
    with col2:
        st.metric("عدد العمليات", len(processes))
    with col3:
        total_time = max([t['finish'] for t in sjf_timeline])
        st.metric("إجمالي الوقت", total_time)

with tab3:
    st.subheader(f"Round Robin (Quantum = {quantum})")
    draw_gantt(rr_timeline, f"Round Robin Gantt Chart (Q={quantum})")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("متوسط وقت الانتظار", f"{rr_wait:.2f}")
    with col2:
        st.metric("عدد العمليات", len(processes))
    with col3:
        total_time = max([t['finish'] for t in rr_timeline])
        st.metric("إجمالي الوقت", total_time)

with tab4:
    st.subheader("📊 مقارنة الخوارزميات")
    
    comparison_df = pd.DataFrame({
        'الخوارزمية': ['FCFS', 'SJF', 'Round Robin'],
        'متوسط وقت الانتظار': [fcfs_wait, sjf_wait, rr_wait]
    })
    
    fig = px.bar(
        comparison_df,
        x='الخوارزمية',
        y='متوسط وقت الانتظار',
        title='مقارنة متوسط وقت الانتظار للخوارزميات الثلاث',
        color='الخوارزمية',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        text='متوسط وقت الانتظار'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(template='plotly_dark', height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    best_algo = comparison_df.loc[comparison_df['متوسط وقت الانتظار'].idxmin(), 'الخوارزمية']
    st.success(f"🏆 أفضل خوارزمية لهذه البيانات: **{best_algo}**")